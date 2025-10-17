import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SceneTextDataset
from model import Seq2Seq
from vocab_utils import build_vocab
import pandas as pd

# --- Paths ---
train_csv = r"C:\Users\zeusk\Downloads\2121\recognition\train.csv"
test_csv  = r"C:\Users\zeusk\Downloads\2121\recognition\test.csv"
base_folder = r"C:\Users\zeusk\Downloads\2121\recognition"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((32,128)),
    transforms.ToTensor()
])

# --- Vocab ---
char2idx, idx2char = build_vocab([train_csv, test_csv])
vocab_size = len(char2idx)

# --- Datasets + Loaders ---
train_dataset = SceneTextDataset(train_csv, base_folder, transform)
test_dataset  = SceneTextDataset(test_csv, base_folder, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32)

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])

# --- Encode text ---
def encode_text(texts):
    max_len = max(len(t) for t in texts)+2
    encoded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        seq = [char2idx['<SOS>']] + [char2idx[c] for c in t] + [char2idx['<EOS>']]
        encoded[i,:len(seq)] = torch.tensor(seq)
    return encoded

# --- Training ---
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, texts in train_loader:
        imgs = imgs.to(device)
        targets = encode_text(texts).to(device)
        optimizer.zero_grad()
        outputs = model(imgs, targets)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, texts in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs, targets=None, teacher_forcing_ratio=0.0)
        pred_idxs = outputs.argmax(2).cpu()
        for pred_seq, target_text in zip(pred_idxs, texts):
            pred_text = "".join([idx2char[i] for i in pred_seq if i not in [char2idx['<PAD>'],char2idx['<SOS>'],char2idx['<EOS>']]])
            correct += (pred_text==target_text)
            total += 1
print(f"Test Accuracy: {100*correct/total:.2f}%")
