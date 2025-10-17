import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class HindiDataset(Dataset):
    def __init__(self, csv_path, base_folder, transform=None, alphabet=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Language'].str.lower() == 'hindi'].reset_index(drop=True)
        self.base_folder = base_folder
        self.transform = transform
        self.alphabet = alphabet
        self.char2idx = {ch:i for i,ch in enumerate(self.alphabet)}

    def encode_text(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx, 0]
        label = str(self.data.iloc[idx, 1])

        img_path = os.path.join(self.base_folder, img_rel_path.replace('/', os.sep))
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        target = self.encode_text(label)
        target = torch.tensor(target, dtype=torch.long)
        return image, target
