import pandas as pd

def build_vocab(csv_paths):
    chars = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        for txt in df['Text']:
            chars.update(list(txt))
    # Special tokens
    chars = ['<PAD>', '<SOS>', '<EOS>'] + sorted(list(chars))
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for c,i in char2idx.items()}
    return char2idx, idx2char
