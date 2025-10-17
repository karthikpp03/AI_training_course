import torch
from torch.utils.data import DataLoader
from dataset import HindiDataset
from model import CRNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Paths
csv_path = r"E:\AI_training_course\capstone_project\data\recognition\train.csv"
base_folder = r"E:\AI_training_course\capstone_project\data\recognition"

# Characters (example Hindi)
alphabet = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह "

# Transform
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset & Loader
dataset = HindiDataset(csv_path, base_folder, transform, alphabet)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=lambda x: x)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(alphabet) + 1
model = CRNN(img_height=32, num_channels=1, num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CTCLoss(blank=num_classes-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for batch in loader:
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets).to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        preds_log_softmax = nn.functional.log_softmax(preds, dim=2)

        # Correct input lengths
        seq_len = preds.size(0)
        input_lengths = torch.full(size=(imgs.size(0),), fill_value=seq_len, dtype=torch.long)

        loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done, loss: {loss.item():.4f}")

torch.save(model.state_dict(), "crnn_hindi.pth")
