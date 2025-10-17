import torch
from model import CRNN
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn


# Character set (same as training)
alphabet = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह "
num_classes = len(alphabet) + 1

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, num_channels=1, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("crnn_hindi.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode(preds):
    # preds: seq_len x batch x num_classes
    preds_idx = preds.argmax(2).permute(1,0)  # batch first: batch x seq_len
    result = ""
    for idx in preds_idx[0]:
        idx = idx.item()
        # Skip repeated characters and blank (last index)
        if idx != len(alphabet) and (len(result) == 0 or alphabet[idx] != result[-1]):
            result += alphabet[idx]
    return result


# Predict
img_path = r"E:\AI_training_course\capstone_project\crops\1.jpg"
image = Image.open(img_path).convert('L')
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(image)
    preds = nn.functional.softmax(preds, dim=2)
    text = decode(preds)
print("Predicted text:", text)
