import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),  # height down
            
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),  # height down
            
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True)
        )
        
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(512, rnn_hidden_size, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(rnn_hidden_size*2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        
        # Adaptive pooling to ensure height = 1
        if h != 1:
            conv = F.adaptive_avg_pool2d(conv, (1, w))
            b, c, h, w = conv.size()
        
        conv = conv.squeeze(2)           # remove height dim
        conv = conv.permute(0, 2, 1)     # batch, width, channels → batch, seq_len, feat
        rnn_out, _ = self.rnn(conv)      # batch, seq_len, hidden*2
        output = self.embedding(rnn_out) # batch, seq_len, num_classes
        output = output.permute(1, 0, 2) # seq_len, batch, num_classes (CTC)
        return output
