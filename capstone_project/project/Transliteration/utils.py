import tensorflow as tf
import re

# Loss function for training
def calc_loss(real, pred):
    # Create mask to ignore padding (zeros)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_obj(real, pred)
    # Only count loss for real words, ignore padding
    loss_ *= tf.cast(mask, dtype=loss_.dtype)
    return tf.reduce_mean(loss_)

# One step of training for a batch
def train_one_batch(encoder, decoder, optimizer, inp, targ, hidden, eng_tokenizer, batch_size):
    with tf.GradientTape() as tape:
        # Pass input through encoder
        enc_output, enc_hidden = encoder(inp, hidden)
        dec_hidden = enc_hidden
        # Prepare decoder input: start token
        start_idx = eng_tokenizer.word_index.get('starttoken', 1)
        dec_input = tf.expand_dims([start_idx] * batch_size, 1)
        
        total_loss = 0
        
        # Loop through each time step in target
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            total_loss += calc_loss(targ[:, t], predictions)
            # Teacher forcing: feed the correct next word
            dec_input = tf.expand_dims(targ[:, t], 1)
    
    # Average loss for the batch
    batch_loss = total_loss / int(targ.shape[1])
    # Compute gradients and update weights
    all_vars = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, all_vars)
    optimizer.apply_gradients(zip(grads, all_vars))
    
    return batch_loss

# Translate a single sentence
def translate_one_sentence(encoder, decoder, sentence, hindi_tokenizer, english_tokenizer, max_len=25, units=512):
    # Clean and lowercase the sentence
    sentence = str(sentence).lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Convert words to numbers
    inputs = [hindi_tokenizer.word_index.get(w, hindi_tokenizer.word_index['<OOV>']) for w in sentence.split()]
    if not inputs:
        return ""
    
    # Pad sequence
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_len, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    # Initialize hidden states
    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    
    # Start token for decoder
    start_idx = english_tokenizer.word_index.get('starttoken', 1)
    dec_input = tf.expand_dims([start_idx], 0)
    
    result = ''
    
    for _ in range(max_len):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = english_tokenizer.index_word.get(predicted_id, '<OOV>')
        
        # Stop at end token
        if word == 'endtoken':
            break
        # Skip start token in output
        if word != 'starttoken':
            result += word + ' '
        # Next input to decoder is predicted word
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result.strip()

# Test a few sentences
def test_example_translations(encoder, decoder, translator, units=512):
    print("\n" + "="*50)
    print("TRANSLATION TEST")
    print("="*50)
    
    sample_sentences = [
        "नमस्ते",
        "धन्यवाद",
        "यह एक परीक्षण वाक्य है",
        "हिंदी से अंग्रेजी अनुवाद",
        "मेरा नाम क्या है"
    ]
    
    for i, sent in enumerate(sample_sentences, 1):
        trans = translate_one_sentence(
            encoder, decoder, sent,
            translator.hindi_tokenizer, translator.english_tokenizer,
            units=units
        )
        print(f"{i}. '{sent}'")
        print(f"   → '{trans}'\n")
