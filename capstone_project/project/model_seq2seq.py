# model_seq2seq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN feature extractor producing (batch, seq_len, feat_dim)
    """
    def __init__(self, in_channels=1, feat_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2,2),                       # h/2
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True),
            nn.MaxPool2d(2,2),                       # h/4
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),               # reduce height
            nn.Conv2d(256,512,3,1,1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),               # further reduce height
            nn.Conv2d(512, feat_dim, 2, 1, 0), nn.ReLU(True)
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        # x: (batch, C, H, W)
        conv = self.cnn(x)           # (batch, feat_dim, h', w')
        b, c, h, w = conv.size()
        if h != 1:
            # ensure height 1
            conv = F.adaptive_avg_pool2d(conv, (1, w))
            b, c, h, w = conv.size()
        conv = conv.squeeze(2)       # (batch, feat_dim, w)
        conv = conv.permute(0,2,1)   # (batch, seq_len=w, feat_dim)
        return conv                  # encoder outputs

class Attention(nn.Module):
    """ Additive (Bahdanau-like) attention """
    def __init__(self, enc_dim, dec_hidden):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, dec_hidden)
        self.dec_proj = nn.Linear(dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden):
        # enc_outputs: (batch, seq_len, enc_dim)
        # dec_hidden: (batch, dec_hidden)
        # returns context: (batch, enc_dim) and attn weights (batch, seq_len)
        enc_e = self.enc_proj(enc_outputs)               # (batch, seq_len, dec_hidden)
        dec_e = self.dec_proj(dec_hidden).unsqueeze(1)   # (batch, 1, dec_hidden)
        e = torch.tanh(enc_e + dec_e)                    # (batch, seq_len, dec_hidden)
        scores = self.v(e).squeeze(-1)                   # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)           # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_dim)
        return context, weights

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_hidden, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attn = Attention(enc_dim, dec_hidden)
        self.rnn = nn.LSTMCell(embed_dim + enc_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, vocab_size)
        self.dec_hidden = dec_hidden

    def forward_step(self, prev_tokens, prev_states, enc_outputs):
        # prev_tokens: (batch,) token indices (last tokens fed)
        # prev_states: (h_t, c_t) each (batch, dec_hidden)
        embedded = self.embedding(prev_tokens)  # (batch, embed_dim)
        h_prev, c_prev = prev_states            # (batch, dec_hidden)

        # attention context
        context, attn_weights = self.attn(enc_outputs, h_prev)  # context: (batch, enc_dim)
        rnn_input = torch.cat([embedded, context], dim=1)
        h_next, c_next = self.rnn(rnn_input, (h_prev, c_prev))
        output_logits = self.out(h_next)         # (batch, vocab_size)
        return output_logits, (h_next, c_next), attn_weights

    def init_states(self, batch_size, device):
        h0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        c0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        return (h0, c0)

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: CNNEncoder, decoder: DecoderWithAttention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, sos_idx=1, eos_idx=2, max_len=128):
        """
        If targets is provided (training), do teacher forcing and return logits for each step.
        images: (batch, C, H, W)
        targets: (batch, seq_len) long (token indices), padded (or None for inference)
        Returns:
            outputs: tensor (batch, max_len, vocab_size) logits (during training we can return all steps)
        """
        device = images.device
        enc_outputs = self.encoder(images)   # (batch, seq_len_enc, enc_dim)
        batch_size = enc_outputs.size(0)
        enc_dim = enc_outputs.size(2)
        vocab_size = self.decoder.out.out_features

        # initialize decoder states
        states = self.decoder.init_states(batch_size, device)
        prev_tokens = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        max_target_len = max_len if targets is None else targets.size(1)
        outputs = []
        for t in range(max_target_len):
            logits, states, attn_weights = self.decoder.forward_step(prev_tokens, states, enc_outputs)
            outputs.append(logits.unsqueeze(1))
            # teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_tokens = targets[:, t].clone().to(device)
            else:
                prev_tokens = logits.argmax(dim=1)
        outputs = torch.cat(outputs, dim=1)  # (batch, max_target_len, vocab)
        return outputs
