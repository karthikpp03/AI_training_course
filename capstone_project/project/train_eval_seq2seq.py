# train_eval_seq2seq.py
import os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_seq2seq import HindiSeqDataset, collate_fn_seq
from model_seq2seq import CNNEncoder, DecoderWithAttention, Seq2SeqModel

# ---------- Config ----------
train_csv = r"C:\Users\zeusk\Downloads\2121\recognition\train.csv"
test_csv  = r"C:\Users\zeusk\Downloads\2121\recognition\test.csv"
base_folder = r"C:\Users\zeusk\Downloads\2121\recognition"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_h, img_w = 32, 256   # make width larger for longer words
batch_size = 16
epochs = 30
teacher_forcing = 0.5
save_every = 5

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------- Dataset & Vocab ----------
# Build vocab from train CSV automatically
train_ds_temp = HindiSeqDataset(train_csv, base_folder, transform=None, build_vocab_from_csv=True)
vocab = train_ds_temp.vocab
print(f"Vocab size (including PAD,SOS,EOS): {len(vocab)}")

train_ds = HindiSeqDataset(train_csv, base_folder, transform=transform, build_vocab_from_csv=False, vocab=vocab)
test_ds  = HindiSeqDataset(test_csv, base_folder, transform=transform, build_vocab_from_csv=False, vocab=vocab)

pad_idx = train_ds.token2idx['<PAD>']
sos_idx = train_ds.token2idx['<SOS>']
eos_idx = train_ds.token2idx['<EOS>']

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn_seq(b, pad_idx))
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn_seq(b, pad_idx))

# ---------- Model ----------
enc = CNNEncoder(in_channels=1, feat_dim=512)
dec = DecoderWithAttention(vocab_size=len(vocab), embed_dim=256, enc_dim=512, dec_hidden=512, padding_idx=pad_idx)
model = Seq2SeqModel(enc, dec).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ---------- Helper decode (greedy) ----------
def greedy_decode(model, image, max_len=120):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(image.to(device), targets=None, teacher_forcing_ratio=0.0, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)
        # outputs: (batch, max_len, vocab)
        preds = outputs.argmax(dim=2)  # (batch, max_len)
        # convert indices to tokens until EOS
        preds = preds[0].tolist()
        tokens = []
        for idx in preds:
            if idx == eos_idx:
                break
            if idx == pad_idx or idx == sos_idx:
                continue
            tokens.append(vocab[idx])
        return ''.join(tokens)

# ---------- Training loop ----------
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for imgs, targets, _ in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)  # (batch, T)
        optimizer.zero_grad()
        outputs = model(imgs, targets=targets, teacher_forcing_ratio=teacher_forcing, sos_idx=sos_idx, eos_idx=eos_idx)
        # outputs: (batch, T, vocab)
        # shift targets: we predict tokens at positions 1..T corresponding to targets[:,1:]
        batch_size, T, V = outputs.size()
        outputs_flat = outputs.view(-1, V)          # (batch*T, V)
        # target tokens to predict: targets[:, 1:(T)] (since targets include SOS at 0)
        target_gold = targets[:, :T]                # ensure shapes align; targets includes SOS and EOS
        target_flat = target_gold.contiguous().view(-1)
        loss = criterion(outputs_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f}")

    # evaluation every few epochs
    if epoch % save_every == 0 or epoch == epochs:
        torch.save(model.state_dict(), f"seq2seq_hindi_epoch{epoch}.pth")
        # quick eval on test set
        model.eval()
        total_chars = 0
        correct_chars = 0
        with torch.no_grad():
            for imgs, targets, raw_text in test_loader:
                pred = greedy_decode(model, imgs)
                gt = raw_text[0]
                total_chars += len(gt)
                # simple char-level comparison
                for i, ch in enumerate(gt):
                    if i < len(pred) and pred[i] == ch:
                        correct_chars += 1
        acc = 0.0 if total_chars==0 else (correct_chars/total_chars)*100
        print(f" -> Saved epoch {epoch}. Test char-accuracy: {acc:.2f}%")
