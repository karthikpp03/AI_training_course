# dataset_seq2seq_simple.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# This class helps us load Hindi text and image pairs from a CSV file
class HindiSeqDataset(Dataset):
    def __init__(self, csv_path, base_folder, transform=None, build_vocab_from_csv=False, vocab=None):
        """
        csv_path: location of the CSV file
        base_folder: where all images are stored
        transform: image transforms (like resize, normalize, etc.)
        build_vocab_from_csv: if True, make vocabulary from all text in CSV
        vocab: if you already have vocab, pass it here
        """

        # read csv file
        self.data = pd.read_csv(csv_path)

        # some csv might have different languages, so keep only hindi rows
        if 'Language' in self.data.columns:
            self.data = self.data[self.data['Language'].str.lower() == 'hindi'].reset_index(drop=True)

        self.base_folder = base_folder
        self.transform = transform

        # if we want to build vocab from scratch
        if build_vocab_from_csv:
            chars = set()
            # take every character from the "Text" column
            for txt in self.data['Text'].astype(str):
                chars.update(list(txt))

            # make tokens (special tokens first)
            # 0 -> <PAD>, 1 -> <SOS>, 2 -> <EOS>
            tokens = ['<PAD>', '<SOS>', '<EOS>'] + sorted(list(chars))
            self.vocab = tokens
        else:
            # make sure vocab is given if not building one
            assert vocab is not None, "Either give vocab or set build_vocab_from_csv=True"
            self.vocab = vocab

        # create dictionary for token to index and back
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def text_to_indices(self, text):
        # convert each character into its index number
        # skip characters not in vocab
        return [self.token2idx[ch] for ch in text if ch in self.token2idx]

    def __len__(self):
        # total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # get one row from the csv using the index
        rel_path = self.data.iloc[idx, 0]   # path like train/hindi/img_123.jpg
        text = str(self.data.iloc[idx, 1])

        # build the full image path
        img_path = os.path.join(self.base_folder, rel_path.replace('/', os.sep))

        # open the image and convert to grayscale
        img = Image.open(img_path).convert('L')

        # apply transformations if given (like resize, normalize)
        if self.transform:
            img = self.transform(img)

        # convert text to list of token ids
        token_ids = self.text_to_indices(text)

        # return the image tensor, tokens, and original text
        return img, torch.tensor(token_ids, dtype=torch.long), text


# This function helps to combine multiple samples into a batch for training
def collate_fn_seq(batch, pad_idx=0):
    """
    batch: list of (image, token_tensor, text)
    Returns:
       imgs: all images stacked together
       targets: padded token sequences with <SOS> and <EOS>
       texts: original text strings
    """
    # unzip the batch into separate lists
    imgs, tokens, texts = zip(*batch)

    # combine all images into a single tensor (batch size x C x H x W)
    imgs = torch.stack(imgs)

    # find the longest sequence length (plus 2 for SOS and EOS)
    max_len = max([t.size(0) for t in tokens]) + 2

    padded = []
    for t in tokens:
        # make a tensor filled with <PAD>
        seq = torch.full((max_len,), pad_idx, dtype=torch.long)

        # add <SOS> at start and <EOS> at end
        seq[0] = 1  # SOS index = 1
        seq[1:1+t.size(0)] = t
        seq[1+t.size(0)] = 2  # EOS index = 2

        padded.append(seq)

    # stack all sequences together
    targets = torch.stack(padded)

    return imgs, targets, list(texts)
