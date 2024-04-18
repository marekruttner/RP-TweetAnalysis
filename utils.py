import re
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    Cleans the text by removing URLs, HTML tags, and reduces multiple spaces to a single space.
    Does not tokenize or remove stop words as model benefits from more contextual information.

    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def compute_metrics(y_true, y_pred):
    """
    Computes MSE, RMSE, and MAE between true and predicted values.

    Parameters:
    - y_true (array-like): Actual values.
    - y_pred (array-like): Model's predictions.

    Returns:
    - dict: Dictionary with MSE, RMSE, and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse.item(), 'MAE': mae}  # rmse.item() to convert from tensor to float
