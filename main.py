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
wandb.init(project="tweet-popularity-prediction", entity="your_wandb_username", config={
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


if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    df['Content'] = df['Content'].apply(preprocess_text)

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

    model = TweetPopularityModel(config.vocab_size, config.embedding_dim, config.hidden_dim,
                                 learning_rate=config.learning_rate)


    wandb_logger = WandbLogger(project="tweet-popularity-prediction", entity="your_wandb_username", log_model="all")

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        gpus=-1,
        callbacks=[ModelCheckpoint(monitor='val_loss')],
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

    wandb_logger.experiment.finish()
