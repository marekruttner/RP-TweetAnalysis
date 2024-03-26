import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming preprocess_text and other utility functions are defined elsewhere
from utils import preprocess_text

# Dataset class
class TweetsDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.scores[idx]

# Model class
class TweetPopularityModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Adjust for bidirectional

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = F.mse_loss(predictions, scores)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, scores = batch
        predictions = self.forward(texts).squeeze(1)
        loss = F.mse_loss(predictions, scores)
        self.log('val_loss', loss, prog_bar=True)
        return loss

# Main execution
if __name__ == "__main__":
    # Load and preprocess your dataset
    df = pd.read_csv('/path/to/your/dataset.csv')
    df['Content'] = df['Content'].apply(preprocess_text)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df['Content'], df['Popularity_Score'], test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_val = vectorizer.transform(X_val).toarray()

    # Create dataset and dataloader
    train_dataset = TweetsDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.float))
    val_dataset = TweetsDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val.values, dtype=torch.float))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Define model and trainer
    model = TweetPopularityModel(vocab_size=5000, embedding_dim=100, hidden_dim=256)
    trainer = pl.Trainer(max_epochs=10, gpus=-1, callbacks=[ModelCheckpoint(monitor='val_loss')])

    # Train the model
    trainer.fit(model, train_loader, val_loader)
