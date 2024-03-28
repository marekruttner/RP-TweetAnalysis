import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download necessary NLTK datasets (only needs to be done once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the WordNet Lemmatizer and load stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses the text by removing URLs, HTML tags, mentions, hashtags, special characters,
    and then applies lowercasing, stopword removal, and lemmatization.

    Parameters:
    - text (str): The text to be cleaned and processed.

    Returns:
    - list[str]: The processed text as a list of words (tokens).
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove mentions (@) and hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters and numbers, keeping only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text into words
    words = text.split()

    # Remove stopwords and lemmatize each word
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return processed_words
  # Return list of words (tokens) instead of a string

class Vocabulary:
    """A simple vocabulary class to map tokens to numerical ids."""
    def __init__(self, max_size=None):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.max_size = max_size

    def build_vocab(self, sentences):
        token_freqs = Counter(token for sent in sentences for token in sent)
        most_common = token_freqs.most_common(self.max_size)
        for idx, (token, _) in enumerate(most_common, start=len(self.token_to_id)):
            self.token_to_id[token] = idx

    def numericalize(self, text):
        return [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in text]

def pad_sequences(sequences, max_length):
    """Pads or truncates a list of sequences to a fixed length."""
    padded_sequences = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded_sequences[i, :length] = torch.tensor(seq[:max_length], dtype=torch.long)
    return padded_sequences

# Example Usage:
# vocab = Vocabulary(max_size=10000)
# vocab.build_vocab([preprocess_text(sent) for sent in all_sentences])
# numericalized_texts = [vocab.numericalize(preprocess_text(sent)) for sent in all_sentences]
# padded_texts = pad_sequences(numericalized_texts, max_length=50)
