import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import preprocess_text

import wandb

# Initialize wandb
wandb.init(project="tweet-popularity-prediction", config={
    "learning_rate": 0.001,
    "architecture": "LSTM",
    "dataset": "dataset.csv",
    "epochs": 10,
    "vocab_size": 5000,
    "embedding_dim": 100,
    "hidden_dim": 256,
})

config = wandb.config


class TweetsDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.scores[idx]

"""
class TweetPopularityModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return output

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = nn.functional.mse_loss(predictions, scores)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = nn.functional.mse_loss(predictions, scores)
        self.log('val_loss', loss, prog_bar=True)
        return loss
"""
class TweetPopularityModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 because of bidirectional

    def forward(self, x):
        # Pass the input through the LSTM layer
        _, (hidden, _) = self.lstm(x)

        # Handling for a bidirectional LSTM
        # Reshape hidden state assuming single layer bidirectional LSTM
        # hidden shape is [num_layers * num_directions, batch, hidden_size]
        # We reshape it to [batch, num_layers * num_directions * hidden_size]
        # For a single layer bidirectional LSTM, it becomes [batch, 2 * hidden_size]
        hidden = hidden.view(1, -1, self.hparams.hidden_dim * 2)

        # For a single layer, you might directly use hidden states
        # But here we reshape to ensure compatibility and clear understanding
        output = self.fc(hidden.squeeze(0))
        return output

    def configure_optimizers(self):
        # Optimizer
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        # Training step
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = nn.functional.mse_loss(predictions, scores)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = nn.functional.mse_loss(predictions, scores)
        self.log('val_loss', loss, prog_bar=True)
        return loss

if __name__ == "__main__":
    # Load and preprocess your dataset
    df = pd.read_csv('dataset.csv')
    df['Content'] = df['Content'].apply(preprocess_text)

    # Example calculation for 'Popularity_Score'
    # Adjust this calculation based on your actual formula
    df['Popularity_Score'] = df['Likes'] + df['Retweets']  # Example calculation

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df['Content'], df['Popularity_Score'], test_size=0.2,
                                                      random_state=42)

    vectorizer = TfidfVectorizer(max_features=config.vocab_size)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_val = vectorizer.transform(X_val).toarray()

    train_dataset = TweetsDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TweetsDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = TweetPopularityModel(input_dim=config.vocab_size, hidden_dim=config.hidden_dim, learning_rate=config.learning_rate)



    wandb_logger = WandbLogger(project="tweet-popularity-prediction", log_model="all")

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        #gpus=1,
        callbacks=[ModelCheckpoint(monitor='val_loss')],
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
