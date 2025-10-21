import tensorflow as tf
from dataset import Translator  
from model import build_model   # Encoder-Decoder with attention
from utils import train_one_batch, test_example_translations
import os


# Configurations

BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 20

print("Starting Hindi-English translation training...")


# Load and prepare data

translator = Translator(max_len=25)
hindi_sentences, english_sentences = translator.load_data('/content/data/hindi_english_parallel.csv')

if not hindi_sentences:
    print("No data loaded. Exiting.")
    exit()

# Make tokenizers and clean text
hindi_sentences, english_sentences = translator.make_tokenizers(hindi_sentences, english_sentences)

if not hindi_sentences:
    print("Tokenization failed. Exiting.")
    exit()

# Convert sentences to sequences of numbers
hindi_seq, english_seq = translator.text_to_seq(hindi_sentences, english_sentences)

# Get vocabulary sizes
h_vocab_size, e_vocab_size = translator.vocab_size()
print(f"Training with {len(hindi_sentences)} samples")
print(f"Hindi vocab: {h_vocab_size}, English vocab: {e_vocab_size}")


# Create TensorFlow dataset

dataset = tf.data.Dataset.from_tensor_slices((hindi_seq, english_seq))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# Build the model

encoder, decoder = build_model(h_vocab_size, e_vocab_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Training loop

print("Training started...")

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    enc_hidden = encoder.init_hidden()
    
    for batch, (inp, targ) in enumerate(dataset):
        batch_loss = train_one_batch(encoder, decoder, optimizer, inp, targ, enc_hidden, translator.english_tokenizer, BATCH_SIZE)
        total_loss += batch_loss
        num_batches += 1
        
        if batch % 50 == 0:
            print(f'   Batch {batch} Loss: {batch_loss.numpy():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f'Epoch {epoch+1}/{EPOCHS} Average Loss: {avg_loss.numpy():.4f}')
    
    # Test every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"\n🔍 Testing after epoch {epoch+1}...")
        test_example_translations(encoder, decoder, translator, UNITS)
        
        # Save checkpoint weights
        encoder.save_weights(f'encoder_epoch_{epoch+1}.h5')
        decoder.save_weights(f'decoder_epoch_{epoch+1}.h5')
        print(f'Checkpoint saved at epoch {epoch+1}')


# Final save

encoder.save_weights('encoder_final.h5')
decoder.save_weights('decoder_final.h5')
translator.save_tokenizers()

print("Training completed!")
print("Final models and tokenizers saved!")
