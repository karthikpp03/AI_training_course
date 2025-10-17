# test_seq2seq_simple.py
# This script tests a trained seq2seq model on an image
# (beginner-friendly version with simple comments)

import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from model_seq2seq import CNNEncoder, DecoderWithAttention, Seq2SeqModel
from dataset_seq2seq import HindiSeqDataset


# PATHS

checkpoint = "seq2seq_best_model.pth"  # trained model file
train_csv = r"E:\AI_training_course\capstone_project\data\recognition\train.csv"
base_folder = r"E:\AI_training_course\capstone_project\data\recognition"


# BUILD VOCAB

# We need vocab exactly same as training
temp_dataset = HindiSeqDataset(train_csv, base_folder, transform=None, build_vocab_from_csv=True)
vocab = temp_dataset.vocab
pad_idx = temp_dataset.token2idx['<PAD>']
sos_idx = temp_dataset.token2idx['<SOS>']
eos_idx = temp_dataset.token2idx['<EOS>']


# LOAD MODEL

encoder = CNNEncoder(in_channels=1, feat_dim=512)
decoder = DecoderWithAttention(vocab_size=len(vocab), embed_dim=256, enc_dim=512, dec_hidden=512, padding_idx=pad_idx)
model = Seq2SeqModel(encoder, decoder)

# load weights
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
model.eval()  # set in evaluation mode


# IMAGE TRANSFORM

transform = transforms.Compose([
    transforms.Resize((32, 320)),  # make image same size as training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# BEAM SEARCH DECODER

def beam_search_decode(model, image, beam_width=5, max_len=120):
    """
    This tries multiple sequences to pick the most likely output
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # get features from encoder
        enc_outputs = model.encoder(image.to(device))
        batch_size = enc_outputs.size(0)

        # start decoder states
        states = model.decoder.init_states(batch_size, device)

        # initialize beam sequences
        beams = [([sos_idx], 0.0, states)]

        # decode step by step
        for step in range(max_len):
            new_beams = []

            for seq, score, states in beams:
                # if EOS already generated, keep as is
                if seq[-1] == eos_idx:
                    new_beams.append((seq, score, states))
                    continue

                prev_token = torch.tensor([seq[-1]], device=device)
                logits, new_states, _ = model.decoder.forward_step(prev_token, states, enc_outputs)
                probs = F.softmax(logits, dim=-1)

                # take top k tokens
                topk_probs, topk_indices = torch.topk(probs[0], beam_width)

                # create new sequences for each top token
                for i in range(beam_width):
                    new_seq = seq + [topk_indices[i].item()]
                    new_score = score + torch.log(topk_probs[i]).item()
                    new_beams.append((new_seq, new_score, new_states))

            # keep only best beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # stop if all beams ended with EOS
            if all(beam[0][-1] == eos_idx for beam in beams):
                break

        # take best sequence and convert to characters
        best_sequence = beams[0][0]
        tokens = []
        for idx in best_sequence[1:]:  # skip SOS
            if idx == eos_idx:
                break
            if idx != pad_idx:
                tokens.append(vocab[idx])

        return ''.join(tokens)



# SIMPLE INFER FUNCTION

def infer(image_path, use_beam_search=True):
    # load image and apply transforms
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = img.to(device)

    if use_beam_search:
        return beam_search_decode(model, img, beam_width=3)
    else:
        # greedy decoding (pick max probability at each step)
        with torch.no_grad():
            outputs = model(img, targets=None, teacher_forcing_ratio=0.0,
                            sos_idx=sos_idx, eos_idx=eos_idx, max_len=120)
            preds = outputs.argmax(dim=2)[0].cpu().tolist()
            result = []
            for idx in preds:
                if idx == eos_idx:
                    break
                if idx == pad_idx or idx == sos_idx:
                    continue
                result.append(vocab[idx])
            return ''.join(result)



# EXAMPLE USAGE

img_path = r"C:\Users\zeusk\Downloads\1212.webp"

print("Predicted (Beam Search):", infer(img_path, use_beam_search=True))
print("Predicted (Greedy):", infer(img_path, use_beam_search=False))
