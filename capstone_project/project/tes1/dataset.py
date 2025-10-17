import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class SceneTextDataset(Dataset):
    def __init__(self, csv_path, base_folder, transform=None):
        self.data = pd.read_csv(csv_path)
        self.base_folder = base_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_folder, row['Filepath'])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        text = row['Text']
        return image, text
