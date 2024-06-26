import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK datasets (only needs to be done once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Cleans and preprocesses the text by removing URLs, HTML tags, mentions, hashtags, special characters,
    and then applies lowercasing, stopword removal, and lemmatization.

    Parameters:
    - text (str): The text to be cleaned and processed.

    Returns:
    - str: The processed text.
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

    # Tokenize the text
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Reconstruct the text from processed words
    processed_text = ' '.join(lemmatized_words)

    return processed_text


def compute_metrics(y_true, y_pred):
    """
    Calculates and returns regression metrics between the true and predicted values.

    Parameters:
    - y_true (array-like): The ground truth target values.
    - y_pred (array-like): The predicted values from the model.

    Returns:
    - dict: A dictionary containing the MSE, RMSE, and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def tokenize_and_pad(text_list, tokenizer, max_length):
    """
    Tokenizes and pads/truncates the list of text to a specified max length.

    Parameters:
    - text_list (list of str): The list of texts to be tokenized and padded.
    - tokenizer: The tokenizer to be used, should have a tokenize method.
    - max_length (int): The maximum length of the tokenized output.

    Returns:
    - torch.Tensor: A tensor of tokenized and padded indices.
    """
    tokenized_texts = [tokenizer.tokenize(text) for text in text_list]
    padded_tokens = torch.tensor([pad_or_truncate(tokens, max_length) for tokens in tokenized_texts])
    return padded_tokens

def pad_or_truncate(tokens, max_length):
    """
    Pads or truncates a list of token ids to a specified maximum length.

    Parameters:
    - tokens (list of int): The token ids to be padded or truncated.
    - max_length (int): The maximum length.

    Returns:
    - list of int: The adjusted list of token ids.
    """
    if len(tokens) > max_length:
        return tokens[:max_length]
    else:
        return tokens + [0] * (max_length - len(tokens))
