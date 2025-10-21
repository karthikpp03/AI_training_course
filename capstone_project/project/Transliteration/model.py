# Import tensorflow and layers
import tensorflow as tf
from tensorflow.keras import layers

# Simple attention layer
class Attention(layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        # These are learnable weight layers
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: decoder hidden state
        # values: encoder output
        query_expanded = tf.expand_dims(query, 1)  # make it compatible for addition
        # Calculate attention score
        score = self.V(tf.nn.tanh(self.W1(query_expanded) + self.W2(values)))
        # Convert scores to probabilities
        attn_weights = tf.nn.softmax(score, axis=1)
        # Multiply attention weights with encoder output
        context = attn_weights * values
        # Sum to get final context vector
        context_vector = tf.reduce_sum(context, axis=1)
        return context_vector, attn_weights

# Encoder LSTM
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embed_dim)  # word embedding
        # LSTM layer
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True, 
                                recurrent_initializer='glorot_uniform')
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, [state_h, state_c]
    
    # Initialize hidden states with zeros
    def init_hidden(self):
        return [tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units))]

# Decoder LSTM with attention
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True,
                                recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)  # final output layer
        self.attention = Attention(units)   # attention layer
    
    def call(self, x, hidden, enc_output):
        # Get context vector from attention
        context_vector, attn_weights = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        # Combine context with embedding
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # Pass through LSTM
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        # Flatten output for final dense layer
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, [state_h, state_c], attn_weights

# Function to build encoder and decoder models
def build_model(h_vocab_size, e_vocab_size, embed_dim=256, units=512, batch_size=64):
    enc = Encoder(h_vocab_size, embed_dim, units, batch_size)
    dec = Decoder(e_vocab_size, embed_dim, units, batch_size)
    return enc, dec
