
# This file has a CNN encoder + attention-based decoder model for Hindi text recognition


import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN ENCODER

class CNNEncoder(nn.Module):
    """
    This part takes an image and turns it into a sequence of features.
    (Think of it like extracting important patterns from the image.)
    """
    def __init__(self, in_channels=1, feat_dim=512):
        super().__init__()
        
        # we use a bunch of conv layers, batchnorm, pooling and dropout
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # reduce height more
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout2d(0.3),

            # Final block to get feature dimension
            nn.Conv2d(512, feat_dim, 2, 1, 0), nn.ReLU(True),
            nn.BatchNorm2d(feat_dim),
            nn.Dropout2d(0.3)
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        # input: (batch, channel, height, width)
        conv = self.cnn(x)  # (batch, feat_dim, h', w')
        b, c, h, w = conv.size()

        # sometimes height may not be 1, so we squeeze it using average pooling
        if h != 1:
            conv = F.adaptive_avg_pool2d(conv, (1, w))
            b, c, h, w = conv.size()

        # remove height dimension and make it (batch, seq_len, feat_dim)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # swap seq_len and feat_dim

        return conv



# ATTENTION MODULE

class Attention(nn.Module):
    """
    This helps the decoder focus on important parts of the encoder output
    (so it doesn't treat all image parts equally)
    """
    def __init__(self, enc_dim, dec_hidden):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, dec_hidden)
        self.dec_proj = nn.Linear(dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, enc_outputs, dec_hidden):
        # enc_outputs: (batch, seq_len, enc_dim)
        # dec_hidden: (batch, dec_hidden)

        # combine encoder outputs and decoder state to get attention weights
        enc_e = self.enc_proj(enc_outputs)
        dec_e = self.dec_proj(dec_hidden).unsqueeze(1)
        e = self.tanh(enc_e + dec_e)

        # compute attention scores
        scores = self.v(e).squeeze(-1)
        weights = torch.softmax(scores, dim=1)  # normalize across seq_len

        # weighted sum (context vector)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights



# DECODER WITH ATTENTION

class DecoderWithAttention(nn.Module):
    """
    This takes the features and generates output characters one by one.
    It uses attention + LSTM + embeddings.
    """
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_hidden, padding_idx):
        super().__init__()

        # word/char embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # attention layer
        self.attn = Attention(enc_dim, dec_hidden)

        # LSTM cell that processes one step at a time
        self.rnn = nn.LSTMCell(embed_dim + enc_dim, dec_hidden)

        # add some normalization and dropout to avoid overfitting
        self.layer_norm = nn.LayerNorm(dec_hidden)
        self.dropout = nn.Dropout(0.3)

        # final layer to get prediction for each vocab token
        self.out = nn.Linear(dec_hidden, vocab_size)
        self.dec_hidden = dec_hidden

    def forward_step(self, prev_tokens, prev_states, enc_outputs):
        # prev_tokens: (batch,) previous output tokens
        # prev_states: (h, c) from previous LSTM step
        embedded = self.embedding(prev_tokens)  # (batch, embed_dim)
        h_prev, c_prev = prev_states

        # get context vector from attention
        context, attn_weights = self.attn(enc_outputs, h_prev)

        # combine embedding and context before feeding into LSTM
        rnn_input = torch.cat([self.dropout(embedded), context], dim=1)

        # LSTM step
        h_next, c_next = self.rnn(rnn_input, (h_prev, c_prev))

        # normalize and apply dropout
        h_next = self.layer_norm(h_next)
        output_logits = self.out(self.dropout(h_next))

        return output_logits, (h_next, c_next), attn_weights

    def init_states(self, batch_size, device):
        # start hidden and cell states as zeros
        h0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        c0 = torch.zeros(batch_size, self.dec_hidden, device=device)
        return (h0, c0)



# FULL SEQ2SEQ MODEL

class Seq2SeqModel(nn.Module):
    """
    This connects the encoder and decoder together.
    Encoder turns image -> features
    Decoder turns features -> predicted text
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, sos_idx=1, eos_idx=2, max_len=128):
        device = images.device

        # run encoder
        enc_outputs = self.encoder(images)
        batch_size = enc_outputs.size(0)
        vocab_size = self.decoder.out.out_features

        # start with empty decoder states
        states = self.decoder.init_states(batch_size, device)

        # all sequences start with <SOS>
        prev_tokens = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        # decide how long to run
        max_target_len = max_len if targets is None else targets.size(1)

        outputs = []

        for t in range(max_target_len):
            # run decoder for one step
            logits, states, attn_weights = self.decoder.forward_step(prev_tokens, states, enc_outputs)
            outputs.append(logits.unsqueeze(1))

            # use teacher forcing sometimes (helps model learn faster)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_tokens = targets[:, t].clone().to(device)
            else:
                prev_tokens = logits.argmax(dim=1)

        # combine outputs for all time steps
        outputs = torch.cat(outputs, dim=1)  # (batch, max_target_len, vocab_size)
        return outputs
