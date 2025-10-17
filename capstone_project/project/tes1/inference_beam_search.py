import torch
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = ['<PAD>', '<SOS>', '<EOS>'] + list("अआइईउऊऋएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह0123456789")
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for i,c in enumerate(vocab)}

transform = transforms.Compose([
    transforms.Resize((32,128)),
    transforms.ToTensor()
])

encoder = EncoderCNN().to(device)
decoder = DecoderRNN(output_dim=len(vocab), enc_hid_dim=512, dec_hid_dim=512).to(device)

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))
encoder.eval()
decoder.eval()

def beam_search(img_path, beam_width=5, max_len=50):
    image = Image.open(img_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    enc_out = encoder(image)
    sequences = [([char2idx['<SOS>']], 0.0, (torch.zeros(1,1,512).to(device), torch.zeros(1,1,512).to(device)))]  # seq, score, hidden
    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden in sequences:
            dec_input = torch.tensor([seq[-1]]).to(device)
            preds, hidden_new, _ = decoder(dec_input, hidden, enc_out)
            probs = torch.log_softmax(preds, dim=1)
            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                candidate = (seq + [topk.indices[0,i].item()], score + topk.values[0,i].item(), hidden_new)
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        # stop if all sequences end with <EOS>
        if all([s[0][-1]==char2idx['<EOS>'] for s in sequences]):
            break
    best_seq = sequences[0][0]
    text = "".join([idx2char[i] for i in best_seq if i not in [char2idx['<SOS>'], char2idx['<EOS>'], char2idx['<PAD>']]])
    return text

# Example:
print(beam_search("E:/AI_training_course/capstone_project/data/recognition/test/hindi/A_image_3455_45.jpg"))
