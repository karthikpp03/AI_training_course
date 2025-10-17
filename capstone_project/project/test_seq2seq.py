# test_seq2seq.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_seq2seq import CNNEncoder, DecoderWithAttention, Seq2SeqModel
from dataset_seq2seq import HindiSeqDataset

# Paths (modify)
checkpoint = "seq2seq_hindi_epoch30.pth"  # change to your saved model
vocab_fileless_build = False
train_csv = r"E:\AI_training_course\capstone_project\data\recognition\train.csv"
base_folder = r"E:\AI_training_course\capstone_project\data\recognition"

# Build vocab same as training — easiest is to build from train CSV like training script did:
temp = HindiSeqDataset(train_csv, base_folder, transform=None, build_vocab_from_csv=True)
vocab = temp.vocab

# load model
enc = CNNEncoder(in_channels=1, feat_dim=512)
dec = DecoderWithAttention(vocab_size=len(vocab), embed_dim=256, enc_dim=512, dec_hidden=512, padding_idx=0)
model = Seq2SeqModel(enc, dec)
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
model.eval()

transform = transforms.Compose([transforms.Resize((32,256)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

def infer(image_path, max_len=120):
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)  # batch 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img, targets=None, teacher_forcing_ratio=0.0, sos_idx=1, eos_idx=2, max_len=max_len)
        preds = outputs.argmax(dim=2)[0].cpu().tolist()
        result = []
        for idx in preds:
            if idx == 2:  # EOS
                break
            if idx == 0 or idx == 1:
                continue
            result.append(vocab[idx])
    return ''.join(result)

# Example usage:
img_path = r"E:\AI_training_course\capstone_project\data\recognition\test\hindi\A_image_3455_45.jpg"
print("Predicted:", infer(img_path))
