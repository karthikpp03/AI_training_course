import tensorflow as tf
from utils import translate_one_sentence, test_example_translations
from dataset import Translator
from model import build_model

print("Loading trained model for testing...")


# Initialize translator and load tokenizers

translator = Translator()
translator.load_tokenizers()  # load saved Hindi & English tokenizers

# Get vocab sizes
h_vocab_size, e_vocab_size = translator.vocab_size()


# Build encoder-decoder model for inference

encoder, decoder = build_model(h_vocab_size, e_vocab_size, batch_size=1)  # batch_size=1 for testing one sentence

# Load trained weights
encoder.load_weights('encoder_final.h5')
decoder.load_weights('decoder_final.h5')

print("Model loaded successfully!")


# Test a few sentences automatically

test_example_translations(encoder, decoder, translator)

# Interactive testing (user input)

print("\n Interactive Testing (type 'quit' to exit):")
while True:
    hindi_sentence = input("\nEnter Hindi sentence: ").strip()
    if hindi_sentence.lower() == 'quit':
        break
    translation = translate_one_sentence(
        encoder, decoder, hindi_sentence,
        translator.hindi_tokenizer, translator.english_tokenizer
    )
    print(f"Translation: {translation}")
