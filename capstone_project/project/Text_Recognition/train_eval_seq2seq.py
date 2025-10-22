# train_seq2seq_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_seq2seq import HindiSeqDataset, collate_fn_seq
from model_seq2seq import CNNEncoder, DecoderWithAttention, Seq2SeqModel

# --- CONFIG ---
train_csv = r"C:\Users\zeusk\Downloads\2121\recognition\train.csv"
test_csv = r"C:\Users\zeusk\Downloads\2121\recognition\test.csv"
base_folder = r"C:\Users\zeusk\Downloads\2121\recognition"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# basic training setup
img_h, img_w = 32, 320
batch_size = 16
epochs = 50
save_every = 5

# --- IMAGE TRANSFORMS ---
# Doing a bit of random augmentation (helps with generalization)
transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- LABEL SMOOTHING LOSS ---
class LabelSmoothingCrossEntropy(nn.Module):
    """Small twist on cross-entropy to make training smoother"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- DATASET AND VOCAB ---
print(" Building vocab from training data...")
temp_ds = HindiSeqDataset(train_csv, base_folder, transform=None, build_vocab_from_csv=True)
vocab = temp_ds.vocab
print(f" Vocab size (with PAD/SOS/EOS): {len(vocab)}")

# reload datasets with same vocab
train_ds = HindiSeqDataset(train_csv, base_folder, transform=transform, build_vocab_from_csv=False, vocab=vocab)
test_ds = HindiSeqDataset(test_csv, base_folder, transform=transform, build_vocab_from_csv=False, vocab=vocab)

pad_idx = train_ds.token2idx['<PAD>']
sos_idx = train_ds.token2idx['<SOS>']
eos_idx = train_ds.token2idx['<EOS>']

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn_seq(b, pad_idx))
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn_seq(b, pad_idx))

# --- MODEL ---
print(" Building Seq2Seq model...")
encoder = CNNEncoder(in_channels=1, feat_dim=512)
decoder = DecoderWithAttention(
    vocab_size=len(vocab),
    embed_dim=256,
    enc_dim=512,
    dec_hidden=512,
    padding_idx=pad_idx
)
model = Seq2SeqModel(encoder, decoder).to(device)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1
)

# --- HELPER FUNCTIONS ---
def greedy_decode(model, image, max_len=120):
    """Simple decoding — just pick top prediction each step"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(image.to(device), targets=None, teacher_forcing_ratio=0.0,
                        sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)
        preds = outputs.argmax(dim=2)[0].tolist()
        result = []
        for idx in preds:
            if idx == eos_idx:
                break
            if idx not in (pad_idx, sos_idx):
                result.append(vocab[idx])
        return ''.join(result)

def beam_search_decode(model, image, beam_width=5, max_len=120):
    """Try multiple possible sequences and pick the best-scored one"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        enc_out = model.encoder(image.to(device))
        batch_size = enc_out.size(0)
        states = model.decoder.init_states(batch_size, device)
        beams = [([sos_idx], 0.0, states)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, states in beams:
                if seq[-1] == eos_idx:
                    new_beams.append((seq, score, states))
                    continue
                prev_token = torch.tensor([seq[-1]], device=device)
                logits, new_states, _ = model.decoder.forward_step(prev_token, states, enc_out)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_idx = torch.topk(probs[0], beam_width)

                for i in range(beam_width):
                    new_seq = seq + [topk_idx[i].item()]
                    new_score = score + torch.log(topk_probs[i]).item()
                    new_beams.append((new_seq, new_score, new_states))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(b[0][-1] == eos_idx for b in beams):
                break

        best_seq = beams[0][0]
        tokens = [vocab[idx] for idx in best_seq[1:] if idx not in (pad_idx, eos_idx)]
        return ''.join(tokens)

def calculate_cer(pred, gt):
    """Character Error Rate (CER) - lower is better"""
    from difflib import SequenceMatcher
    return 1 - SequenceMatcher(None, gt, pred).ratio()

def evaluate_model(model, test_loader, epoch, use_beam_search=False):
    """Run evaluation after each save"""
    model.eval()
    total_chars, correct_chars, total_cer, total_word_acc, total_samples = 0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, targets, raw_text in test_loader:
            pred = beam_search_decode(model, imgs, beam_width=3) if use_beam_search else greedy_decode(model, imgs)
            gt = raw_text[0]
            total_samples += 1
            total_chars += len(gt)
            correct_chars += sum(1 for i, ch in enumerate(gt) if i < len(pred) and pred[i] == ch)
            total_cer += calculate_cer(pred, gt)
            total_word_acc += 1.0 if pred == gt else 0.0

            if total_samples <= 3:
                print(f"Sample {total_samples}: GT='{gt}', Pred='{pred}', CER={calculate_cer(pred, gt):.3f}")

    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    avg_cer = total_cer / total_samples
    avg_word_acc = (total_word_acc / total_samples) * 100
    print(f" -> Epoch {epoch}: CharAcc={char_acc:.2f}%, CER={avg_cer:.3f}, WordAcc={avg_word_acc:.2f}%")
    return char_acc, avg_cer

# --- TRAINING LOOP ---
print("\n Starting training...\n")
best_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    tf_ratio = max(0.3, 0.9 - (epoch / epochs) * 0.6)  # teacher forcing schedule

    for imgs, targets, _ in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(imgs, targets=targets, teacher_forcing_ratio=tf_ratio,
                        sos_idx=sos_idx, eos_idx=eos_idx)
        bsz, T, V = outputs.size()
        outputs_flat = outputs.view(-1, V)
        targets_flat = targets[:, :T].contiguous().view(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | TF={tf_ratio:.2f} | LR={scheduler.get_last_lr()[0]:.6f}")

    # --- Save checkpoints & evaluate ---
    if epoch % save_every == 0 or epoch == epochs:
        torch.save(model.state_dict(), f"seq2seq_epoch{epoch}.pth")
        char_acc, _ = evaluate_model(model, test_loader, epoch, use_beam_search=False)

        if epoch == epochs:
            print("\n Final Beam Search Evaluation:")
            evaluate_model(model, test_loader, epoch, use_beam_search=True)

        if char_acc > best_acc:
            best_acc = char_acc
            torch.save(model.state_dict(), "seq2seq_best_model.pth")
            print(f"New best model saved! CharAcc={best_acc:.2f}%")

print("\n Training complete!")
print(f" Best character accuracy: {best_acc:.2f}%")
