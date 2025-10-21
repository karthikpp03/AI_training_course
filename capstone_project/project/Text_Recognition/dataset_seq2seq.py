# dataset_seq2seq_simple.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Just a small dataset class to handle Hindi text + images
class HindiSeqDataset(Dataset):
    def __init__(self, csv_path, base_folder, transform=None, build_vocab_from_csv=False, vocab=None):
        """
        csv_path -> path to the csv file
        base_folder -> folder where all images are stored
        transform -> optional image transformations
        build_vocab_from_csv -> True if we wanna create vocab from scratch
        vocab -> if already have vocab, pass that
        """

        # load csv
        self.data = pd.read_csv(csv_path)

        # sometimes dataset might have other languages too, so keep only hindi ones
        if 'Language' in self.data.columns:
            self.data = self.data[self.data['Language'].str.lower() == 'hindi'].reset_index(drop=True)

        self.base_folder = base_folder
        self.transform = transform

        # vocab creation or loading
        if build_vocab_from_csv:
            all_chars = set()
            for txt in self.data['Text'].astype(str):
                all_chars.update(list(txt))  # grab every character
            
            # keeping special tokens at start (0 = PAD, 1 = SOS, 2 = EOS)
            self.vocab = ['<PAD>', '<SOS>', '<EOS>'] + sorted(list(all_chars))
        else:
            # if we’re not building, we need a vocab
            assert vocab is not None, "Give a vocab or set build_vocab_from_csv=True"
            self.vocab = vocab

        # dictionaries for easy conversion
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def text_to_indices(self, text):
        # turn a text string into a list of index numbers
        # ignore anything not in vocab
        return [self.token2idx[ch] for ch in text if ch in self.token2idx]

    def __len__(self):
        # just the number of rows in csv
        return len(self.data)

    def __getitem__(self, idx):
        # pick one row (image + text)
        rel_path = self.data.iloc[idx, 0]
        text = str(self.data.iloc[idx, 1])

        # combine folder + path properly
        img_path = os.path.join(self.base_folder, rel_path.replace('/', os.sep))

        # open and make it grayscale
        img = Image.open(img_path).convert('L')

        # apply any transforms (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        # convert text to indices (numbers)
        token_ids = self.text_to_indices(text)

        # return image tensor, token tensor, and original text
        return img, torch.tensor(token_ids, dtype=torch.long), text


# this fn just prepares a batch properly for seq2seq training
def collate_fn_seq(batch, pad_idx=0):
    """
    batch -> list of (image, token_tensor, text)
    returns:
        imgs -> all stacked image tensors
        targets -> padded token ids with <SOS> and <EOS>
        texts -> raw text
    """
    imgs, tokens, texts = zip(*batch)

    # combine all imgs into a batch tensor
    imgs = torch.stack(imgs)

    # find the max seq length and add 2 (for SOS + EOS)
    max_len = max([t.size(0) for t in tokens]) + 2

    padded_seq = []
    for t in tokens:
        # fill with PAD token
        seq = torch.full((max_len,), pad_idx, dtype=torch.long)
        seq[0] = 1  # <SOS>
        seq[1:1+t.size(0)] = t
        seq[1+t.size(0)] = 2  # <EOS>
        padded_seq.append(seq)

    # stack all seqs
    targets = torch.stack(padded_seq)

    return imgs, targets, list(texts)
