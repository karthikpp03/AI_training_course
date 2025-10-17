# dataset_seq2seq.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HindiSeqDataset(Dataset):
    def __init__(self, csv_path, base_folder, transform=None, build_vocab_from_csv=False, vocab=None):
        """
        If build_vocab_from_csv=True, we will build char-level vocab from CSV texts and return it (user must persist).
        Otherwise pass vocab (list of tokens) explicitly.
        """
        self.data = pd.read_csv(csv_path)
        # keep only hindi rows (some csv contain many languages)
        if 'Language' in self.data.columns:
            self.data = self.data[self.data['Language'].str.lower() == 'hindi'].reset_index(drop=True)
        self.base_folder = base_folder
        self.transform = transform

        if build_vocab_from_csv:
            chars = set()
            for txt in self.data['Text'].astype(str):
                chars.update(list(txt))
            # special tokens order: PAD(0), SOS(1), EOS(2)
            tokens = ['<PAD>', '<SOS>', '<EOS>'] + sorted(list(chars))
            self.vocab = tokens
        else:
            assert vocab is not None, "Provide vocab or set build_vocab_from_csv=True"
            self.vocab = vocab

        # mappings
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def text_to_indices(self, text):
        # convert string to list of token indices (without SOS/EOS here; collate will add)
        return [self.token2idx[ch] for ch in text if ch in self.token2idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel = self.data.iloc[idx, 0]   # path like train/hindi/...
        text = str(self.data.iloc[idx, 1])
        img_path = os.path.join(self.base_folder, rel.replace('/', os.sep))
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        token_ids = self.text_to_indices(text)
        return img, torch.tensor(token_ids, dtype=torch.long), text

def collate_fn_seq(batch, pad_idx=0):
    """
    batch: list of tuples (img_tensor, token_tensor, text_str)
    Returns:
       images: (batch, C, H, W)
       targets_padded: (batch, max_len) with <SOS> and <EOS> added
       raw_texts: list of ground-truth strings
    """
    imgs, tokens, texts = zip(*batch)
    imgs = torch.stack(imgs)
    # prepare targets with SOS and EOS
    max_len = max([t.size(0) for t in tokens]) + 2  # +SOS +EOS
    padded = []
    for t in tokens:
        seq = torch.full((max_len,), pad_idx, dtype=torch.long)
        seq[0] = 1  # SOS index = 1
        seq[1:1+t.size(0)] = t
        seq[1+t.size(0)] = 2  # EOS = 2
        padded.append(seq)
    targets = torch.stack(padded)
    return imgs, targets, list(texts)
