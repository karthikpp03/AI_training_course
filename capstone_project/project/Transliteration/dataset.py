# Import libraries we need
import pandas as pd        # For working with tables / CSV files
import re                  # For cleaning text
import tensorflow as tf    # For tokenizing and padding text
import pickle              # For saving/loading tokenizers

# Simple Hindi to English translator
class Translator:
    def __init__(self, max_words=20000, max_len=25):
        self.max_words = max_words
        self.max_len = max_len
        self.hindi_tokenizer = None
        self.english_tokenizer = None

    # Load CSV and prepare sentences
    def load_data(self, file_path):
        try:
            # Try reading normally
            try:
                data = pd.read_csv(file_path, encoding='utf-8')
            except:
                # If it fails, try tab-separated
                data = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')
            
            print("File loaded!")

            # Make sure columns are named correctly
            if 'hindi' not in data.columns or 'english' not in data.columns:
                if len(data.columns) >= 2:
                    data.columns = ['hindi', 'english'] + list(data.columns[2:])
                    print("Columns renamed to 'hindi' and 'english'")

            # Keep only Hindi and English, remove empty or duplicate rows
            data = data[['hindi', 'english']].dropna().drop_duplicates()
            data = data[(data['hindi'].str.strip() != '') & (data['english'].str.strip() != '')]
            data['hindi'] = data['hindi'].astype(str).str.strip()
            data['english'] = data['english'].astype(str).str.strip()

            # If dataset is big, use only 50k samples
            if len(data) > 50000:
                data = data.sample(50000, random_state=42)
                print("Using 50,000 samples")

            # Filter sentences by length
            data = self.filter_data(data)

            # Add start and end tokens to English
            data['english'] = data['english'].apply(lambda x: 'starttoken ' + x + ' endtoken')

            print(f"Final data size: {len(data)}")
            return data['hindi'].tolist(), data['english'].tolist()

        except Exception as e:
            print("Error loading file:", e)
            return [], []

    # Remove too short or too long sentences
    def filter_data(self, data):
        data['h_len'] = data['hindi'].apply(lambda x: len(str(x).split()))
        data['e_len'] = data['english'].apply(lambda x: len(str(x).split()))
        filtered = data[
            (data['h_len'] >= 2) & (data['h_len'] <= 25) &
            (data['e_len'] >= 2) & (data['e_len'] <= 25)
        ]
        print(f"Filtered {len(data)} -> {len(filtered)}")
        return filtered.drop(['h_len', 'e_len'], axis=1)

    # Clean text: lowercase, remove weird characters, remove extra spaces
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\?\!,]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Make tokenizers for Hindi and English
    def make_tokenizers(self, hindi_list, english_list):
        if not hindi_list or not english_list:
            print("No data for tokenization")
            return [], []

        # Clean the text
        hindi_list = [self.clean_text(t) for t in hindi_list]
        english_list = [self.clean_text(t) for t in english_list]

        # Hindi tokenizer
        self.hindi_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.max_words, filters='', oov_token='<OOV>'
        )
        self.hindi_tokenizer.fit_on_texts(hindi_list)

        # English tokenizer
        self.english_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.max_words, filters='', oov_token='<OOV>'
        )
        self.english_tokenizer.fit_on_texts(english_list)

        # Make sure start/end tokens are there
        self._check_special_tokens()
        
        print(f"Hindi vocab: {len(self.hindi_tokenizer.word_index)}")
        print(f"English vocab: {len(self.english_tokenizer.word_index)}")

        return hindi_list, english_list

    # Make sure special tokens exist
    def _check_special_tokens(self):
        tokens = ['starttoken', 'endtoken', '<OOV>']
        for t in tokens:
            if t not in self.english_tokenizer.word_index:
                idx = len(self.english_tokenizer.word_index) + 1
                self.english_tokenizer.word_index[t] = idx
                self.english_tokenizer.index_word[idx] = t
        print("Special tokens checked")

    # Convert text to numbers and pad sequences
    def text_to_seq(self, hindi_list, english_list):
        h_seq = self.hindi_tokenizer.texts_to_sequences(hindi_list)
        e_seq = self.english_tokenizer.texts_to_sequences(english_list)

        # Pad sequences to same length
        h_pad = tf.keras.preprocessing.sequence.pad_sequences(h_seq, maxlen=self.max_len, padding='post')
        e_pad = tf.keras.preprocessing.sequence.pad_sequences(e_seq, maxlen=self.max_len, padding='post')
        return h_pad, e_pad

    # Get vocab sizes
    def vocab_size(self):
        return len(self.hindi_tokenizer.word_index) + 1, len(self.english_tokenizer.word_index) + 1

    # Save tokenizers for later
    def save_tokenizers(self, h_file='hindi_tokenizer.pickle', e_file='english_tokenizer.pickle'):
        with open(h_file, 'wb') as f:
            pickle.dump(self.hindi_tokenizer, f)
        with open(e_file, 'wb') as f:
            pickle.dump(self.english_tokenizer, f)
        print("Tokenizers saved!")

    # Load saved tokenizers
    def load_tokenizers(self, h_file='hindi_tokenizer.pickle', e_file='english_tokenizer.pickle'):
        with open(h_file, 'rb') as f:
            self.hindi_tokenizer = pickle.load(f)
        with open(e_file, 'rb') as f:
            self.english_tokenizer = pickle.load(f)
        print("Tokenizers loaded!")
