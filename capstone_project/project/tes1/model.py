import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Encoder CNN -----
class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1, out_features=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, out_features, 3, 1, 1), nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # height=1

    def forward(self, x):
        x = self.cnn(x)               # batch x 256 x H' x W'
        x = self.adaptive_pool(x)     # batch x 256 x 1 x W'
        x = x.squeeze(2).permute(0,2,1)  # batch x W' x 256
        return x

# ----- Attention -----
class Attention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + encoder_dim, encoder_dim)
        self.v = nn.Linear(encoder_dim,1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: batch x hidden_dim
        # encoder_outputs: batch x seq_len x encoder_dim
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1,seq_len,1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = F.softmax(self.v(energy), dim=1)
        context = torch.bmm(attention.transpose(1,2), encoder_outputs).squeeze(1)
        return context, attention

# ----- Decoder RNN -----
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, encoder_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = Attention(hidden_dim, encoder_dim)

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5):
        batch_size = encoder_outputs.size(0)
        max_len = targets.size(1) if targets is not None else 50
        hidden = (torch.zeros(1,batch_size,256).to(encoder_outputs.device),
                  torch.zeros(1,batch_size,256).to(encoder_outputs.device))
        inputs = torch.full((batch_size,), 1, dtype=torch.long).to(encoder_outputs.device)  # <SOS>
        outputs = []

        for t in range(max_len):
            embedded = self.embedding(inputs)
            context, _ = self.attention(hidden[0].squeeze(0), encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
            out, hidden = self.rnn(rnn_input, hidden)
            out_vocab = self.fc(out.squeeze(1))
            outputs.append(out_vocab.unsqueeze(1))
            top1 = out_vocab.argmax(1)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inputs = targets[:,t]
            else:
                inputs = top1
        outputs = torch.cat(outputs, dim=1)
        return outputs

# ----- Full Seq2Seq -----
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(vocab_size)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        enc_out = self.encoder(x)
        out = self.decoder(enc_out, targets, teacher_forcing_ratio)
        return out
