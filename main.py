import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming utils.py is in the same directory and contains the necessary functions
from utils import preprocess_text, Vocabulary, pad_sequences

import wandb

# Initialize wandb
wandb.init(project="tweet-popularity-prediction", config={
    "learning_rate": 0.001,
    "architecture": "GRU",
    "dataset": "dataset.csv",
    "epochs": 30,
    "vocab_size": 10000,
    "embedding_dim": 200,
    "hidden_dim": 128,
    "num_layers": 2,
    "batch_size": 64,
    "max_seq_length": 100,
})

config = wandb.config

class TweetsDataset(Dataset):
    def __init__(self, texts, scores, additional_features):
        self.texts = texts
        self.scores = scores
        self.additional_features = additional_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.scores[idx], self.additional_features[idx]


class TweetPopularityModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, additional_feature_dim, learning_rate, num_layers):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim + additional_feature_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, additional_features):
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        combined = torch.cat((embedded, additional_features.unsqueeze(1).repeat(1, embedded.size(1), 1)),
                             dim=2)  # [batch_size, seq_length, embedding_dim + additional_feature_dim]

        # GRU output handling
        _, hidden = self.gru(combined)  # hidden: [num_layers*num_directions, batch_size, hidden_size]
        hidden_fwd = hidden[-2, :, :]  # Forward direction of the last layer
        hidden_bwd = hidden[-1, :, :]  # Backward direction of the last layer
        hidden_cat = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # Concatenate the forward and backward states

        # Ensure the output matches the expected size for the mse_loss calculation
        output = self.fc(hidden_cat)  # [batch_size, 1]
        return output.squeeze()  # Squeeze to [batch_size], matching the scores tensor

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        texts, scores, additional_features = batch
        predictions = self.forward(texts, additional_features)
        loss = nn.functional.mse_loss(predictions, scores)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, scores, additional_features = batch
        predictions = self.forward(texts, additional_features)
        if predictions.dim() > 1:  # Ensure predictions are [batch_size]
            predictions = predictions.squeeze(-1)
        if scores.dim() > 1:  # Ensure scores are [batch_size]
            scores = scores.squeeze(-1)

        print(f"Predictions shape: {predictions.shape}, Scores shape: {scores.shape}")  # Debugging shapes

        loss = nn.functional.mse_loss(predictions, scores)
        self.log('val_loss', loss, prog_bar=True)
        return loss

if __name__ == "__main__":
    # Load and preprocess your dataset
    df = pd.read_csv(config.dataset)
    df['Content'] = df['Content'].apply(preprocess_text)

    # Build vocabulary and prepare sequences
    vocab = Vocabulary(max_size=config.vocab_size)
    vocab.build_vocab(df['Content'].tolist())
    numericalized_texts = [vocab.numericalize(text) for text in df['Content']]
    padded_texts = pad_sequences(numericalized_texts, max_length=config.max_seq_length)

    # Prepare additional features and scores
    df['hour_of_day'] = pd.to_datetime(df['Timestamp']).dt.hour
    additional_features = StandardScaler().fit_transform(df[['hour_of_day']])

    df['Popularity_Score'] = df['Likes'] / df['Analytics']
    scores = scores_tensor = torch.tensor(df['Popularity_Score'].values, dtype=torch.float).view(-1)  # Ensure it's 1D if needed


    # Split data into training and validation sets
    train_texts, val_texts, train_scores, val_scores, train_additional, val_additional = train_test_split(
        padded_texts, scores, additional_features, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TweetsDataset(train_texts, train_scores, torch.tensor(train_additional, dtype=torch.float))
    val_dataset = TweetsDataset(val_texts, val_scores, torch.tensor(val_additional, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize model, logger, and trainer
    model = TweetPopularityModel(vocab_size=config.vocab_size, hidden_dim=config.hidden_dim,
                                 embedding_dim=config.embedding_dim, additional_feature_dim=1,
                                 learning_rate=config.learning_rate, num_layers=config.num_layers)

    wandb_logger = WandbLogger(project="tweet-popularity-prediction", log_model="all")
    trainer = pl.Trainer(max_epochs=config.epochs, callbacks=[ModelCheckpoint(monitor='val_loss')],
                         logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
