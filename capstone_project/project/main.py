import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# -----------------------
# Dataset
# -----------------------
class HindiDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, base_folder, transform=None, alphabet=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Language'].str.lower()=='hindi'].reset_index(drop=True)
        self.base_folder = base_folder
        self.transform = transform
        self.alphabet = alphabet
        self.char2idx = {ch:i for i,ch in enumerate(self.alphabet)}

    def encode_text(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx,0]
        label = str(self.data.iloc[idx,1])
        img_path = os.path.join(self.base_folder, img_rel_path.replace('/', os.sep))
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(self.encode_text(label), dtype=torch.long)
        return image, target, label

# -----------------------
# Model
# -----------------------
class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(inplace=True)
        )
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(512,rnn_hidden_size,bidirectional=True,batch_first=True)
        self.embedding = nn.Linear(rnn_hidden_size*2,num_classes)

    def forward(self,x):
        conv = self.cnn(x)
        b,c,h,w = conv.size()
        if h!=1:
            conv = nn.functional.adaptive_avg_pool2d(conv,(1,w))
            b,c,h,w = conv.size()
        conv = conv.squeeze(2).permute(0,2,1)
        rnn_out,_ = self.rnn(conv)
        output = self.embedding(rnn_out)
        output = output.permute(1,0,2)
        return output

# -----------------------
# Decode function
# -----------------------
def ctc_decode(preds, alphabet):
    preds_idx = preds.argmax(2).permute(1,0)
    result = ""
    for idx in preds_idx[0]:
        idx = idx.item()
        if idx != len(alphabet) and (len(result)==0 or alphabet[idx]!=result[-1]):
            result += alphabet[idx]
    return result

# -----------------------
# Paths & Alphabet
# -----------------------
train_csv = r"E:\AI_training_course\capstone_project\data\recognition\train.csv"
test_csv  = r"E:\AI_training_course\capstone_project\data\recognition\test.csv"
base_folder = r"E:\AI_training_course\capstone_project\data\recognition"
alphabet = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह "

# -----------------------
# Transforms, datasets, loaders
# -----------------------
transform = transforms.Compose([
    transforms.Resize((32,128)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = HindiDataset(train_csv, base_folder, transform, alphabet)
test_dataset = HindiDataset(test_csv, base_folder, transform, alphabet)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x:x)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x)

# -----------------------
# Device, model, optimizer, criterion
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(alphabet)+1
model = CRNN(img_height=32, num_channels=1, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=num_classes-1)

# -----------------------
# Training
# -----------------------
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        imgs, targets, _ = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        targets_lengths = torch.tensor([len(t) for t in targets],dtype=torch.long)
        targets = torch.cat(targets).to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        preds_log_softmax = nn.functional.log_softmax(preds,dim=2)
        input_lengths = torch.full(size=(imgs.size(0),),fill_value=preds.size(0),dtype=torch.long)
        loss = criterion(preds_log_softmax, targets, input_lengths, targets_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} done, avg loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(),"crnn_hindi.pth")

# -----------------------
# Evaluation on test set
# -----------------------
model.eval()
total_chars = 0
correct_chars = 0

with torch.no_grad():
    for batch in test_loader:
        img, _, label = batch[0]
        img = img.unsqueeze(0).to(device)
        preds = model(img)
        preds = nn.functional.softmax(preds, dim=2)
        pred_text = ctc_decode(preds, alphabet)

        total_chars += len(label)
        for i,c in enumerate(label):
            if i < len(pred_text) and c==pred_text[i]:
                correct_chars += 1

accuracy = correct_chars/total_chars*100
print(f"Character-level accuracy on test set: {accuracy:.2f}%")
