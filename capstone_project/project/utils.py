import torch

hindi_alphabet = "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़ड़ढ़फ़ँंःॅॉॆेैोौ् "

class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.dict = {char: i + 1 for i, char in enumerate(alphabet)}

    def encode(self, texts):
        lengths = [len(s) for s in texts]
        joined = ''.join(texts)
        encoded = [self.dict[c] for c in joined if c in self.dict]
        return (torch.tensor(encoded), torch.tensor(lengths))

    def decode(self, preds, lengths):
        texts = []
        index = 0
        for l in lengths:
            text = ''
            for i in range(l):
                c = preds[index]
                if c != 0:
                    text += self.alphabet[c - 1]
                index += 1
            texts.append(text)
        return texts
