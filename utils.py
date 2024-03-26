import re
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def preprocess_text(text):
    """
    Cleans the text by removing URLs, HTML tags, mentions, hashtags, and special characters.
    This function aims to standardize the text input for better model performance.

    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove mentions (@) and hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters and numbers, keeping only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
