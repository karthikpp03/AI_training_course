# test_seq2seq_model.py
# Just testing the trained seq2seq model on a sample image

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model_seq2seq import CNNEncoder, DecoderWithAttention, Seq2SeqModel
from dataset_seq2seq import HindiSeqDataset

# --- Paths and basic setup ---
checkpoint_path = "seq2seq_best_model.pth"   # model checkpoint
train_csv = r"E:\AI_training_course\capstone_project\data\recognition\train.csv"
base_folder = r"E:\AI_training_course\capstone_project\data\recognition"

# --- Build vocab (same as what was used during training) ---
print("Building vocab from training data...")
temp_ds = HindiSeqDataset(train_csv, base_folder, build_vocab_from_csv=True)
vocab = temp_ds.vocab
pad_idx = temp_ds.token2idx['<PAD>']
sos_idx = temp_ds.token2idx['<SOS>']
eos_idx = temp_ds.token2idx['<EOS>']

# --- Load model ---
print("Loading trained Seq2Seq model...")
encoder = CNNEncoder(in_channels=1, feat_dim=512)
decoder = DecoderWithAttention(
    vocab_size=len(vocab),
    embed_dim=256,
    enc_dim=512,
    dec_hidden=512,
    padding_idx=pad_idx
)
model = Seq2SeqModel(encoder, decoder)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((32, 320)),  # resize same as training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Beam Search decoding (slightly fancy way to decode text) ---
def beam_search_decode(model, image, beam_width=5, max_len=120):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        enc_out = model.encoder(image.to(device))
        batch_size = enc_out.size(0)
        states = model.decoder.init_states(batch_size, device)

        beams = [([sos_idx], 0.0, states)]  # (sequence, score, hidden states)

        for step in range(max_len):
            new_beams = []
            for seq, score, states in beams:
                if seq[-1] == eos_idx:  # if already ended, just keep it
                    new_beams.append((seq, score, states))
                    continue

                prev_token = torch.tensor([seq[-1]], device=device)
                logits, new_states, _ = model.decoder.forward_step(prev_token, states, enc_out)
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_idx = torch.topk(probs[0], beam_width)
                for i in range(beam_width):
                    next_seq = seq + [topk_idx[i].item()]
                    next_score = score + torch.log(topk_probs[i]).item()
                    new_beams.append((next_seq, next_score, new_states))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # stop if all beams finished
            if all(b[0][-1] == eos_idx for b in beams):
                break

        # get the best one and convert tokens back to chars
        best_seq = beams[0][0]
        tokens = []
        for idx in best_seq[1:]:  # skip SOS
            if idx == eos_idx:
                break
            if idx not in [pad_idx]:
                tokens.append(vocab[idx])

        return ''.join(tokens)

# --- Simple inference ---
def infer(image_path, use_beam_search=True):
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = img.to(device)

    if use_beam_search:
        return beam_search_decode(model, img, beam_width=3)
    else:
        # Greedy decoding: simple and fast
        with torch.no_grad():
            outputs = model(
                img,
                targets=None,
                teacher_forcing_ratio=0.0,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=120
            )
            preds = outputs.argmax(dim=2)[0].cpu().tolist()
            result = []
            for idx in preds:
                if idx in [pad_idx, sos_idx]:
                    continue
                if idx == eos_idx:
                    break
                result.append(vocab[idx])
            return ''.join(result)

# --- Try on one image ---
img_path = r"C:\Users\zeusk\Downloads\1212.webp"
print(" Running inference on:", img_path)
print(" Beam Search Prediction:", infer(img_path, use_beam_search=True))
print(" Greedy Prediction:", infer(img_path, use_beam_search=False))
