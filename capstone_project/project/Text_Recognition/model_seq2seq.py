# cnn_seq2seq_model.py
# CNN Encoder + Attention Decoder for Hindi text recognition

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- ENCODER ---------- #
# basically takes an image and squeezes out features like CNN usually does
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, feat_dim=512):
        super().__init__()
        # using bunch of conv + pool + dropout layers (kinda standard CNN)
        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # shrink height more
            nn.Dropout2d(0.2),

            # block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout2d(0.3),

            # final layer to get our feature dim
            nn.Conv2d(512, feat_dim, 2, 1, 0), nn.ReLU(True),
            nn.BatchNorm2d(feat_dim),
            nn.Dropout2d(0.3)
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        # x = (batch, channels, height, width)
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        # if height isn't 1, reduce it
        if h != 1:
            conv = F.adaptive_avg_pool2d(conv, (1, w))
            b, c, h, w = conv.size()

        # squeeze height and make (batch, seq_len, feat_dim)
        conv = conv.squeeze(2).permute(0, 2, 1)
        return conv


# ---------- ATTENTION MODULE ---------- #
# helps decoder look at important parts of encoder output
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_hidden):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, dec_hidden)
        self.dec_proj = nn.Linear(dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, enc_outputs, dec_hidden):
        # enc_outputs: (batch, seq_len, enc_dim)
        # dec_hidden: (batch, dec_hidden)
        enc_e = self.enc_proj(enc_outputs)
        dec_e = self.dec_proj(dec_hidden).unsqueeze(1)
        energy = self.tanh(enc_e + dec_e)

        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)  # attention weights
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, weights


# ---------- DECODER WITH ATTENTION ---------- #
# takes features + generates text character by character
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_hidden, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attn = Attention(enc_dim, dec_hidden)
        self.rnn = nn.LSTMCell(embed_dim + enc_dim, dec_hidden)
        self.layer_norm = nn.LayerNorm(dec_hidden)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(dec_hidden, vocab_size)
        self.dec_hidden = dec_hidden

    def forward_step(self, prev_tokens, prev_states, enc_outputs):
        # prev_tokens = (batch,)
        # prev_states = (h, c)
        emb = self.embedding(prev_tokens)
        h_prev, c_prev = prev_states

        # get context vector from attention
        context, attn_weights = self.attn(enc_outputs, h_prev)

        # combine embedding + context before feeding into LSTM
        rnn_input = torch.cat([self.dropout(emb), context], dim=1)
        h_next, c_next = self.rnn(rnn_input, (h_prev, c_prev))

        h_next = self.layer_norm(h_next)
        logits = self.out(self.dropout(h_next))

        return logits, (h_next, c_next), attn_weights

    def init_states(self, batch_size, device):
        # start everything from zero
        h0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        c0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        return (h0, c0)


# ---------- FULL SEQ2SEQ ---------- #
# combines encoder + decoder into one model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5,
                sos_idx=1, eos_idx=2, max_len=128):

        device = images.device
        enc_outputs = self.encoder(images)
        batch_size = enc_outputs.size(0)
        vocab_size = self.decoder.out.out_features

        # init hidden states
        states = self.decoder.init_states(batch_size, device)

        # start with SOS tokens
        prev_tokens = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        # decide max length
        max_target_len = max_len if targets is None else targets.size(1)
        outputs = []

        for t in range(max_target_len):
            logits, states, attn_weights = self.decoder.forward_step(prev_tokens, states, enc_outputs)
            outputs.append(logits.unsqueeze(1))

            # teacher forcing (sometimes use target tokens instead of predictions)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_tokens = targets[:, t].clone().to(device)
            else:
                prev_tokens = logits.argmax(dim=1)

        outputs = torch.cat(outputs, dim=1)
        return outputs
