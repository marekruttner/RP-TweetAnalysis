import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2Tokenizer, AdamW
from utils import preprocess_text, compute_metrics  # Import utility functions

import wandb

# Initialize wandb
wandb.init(project="tweet-popularity-prediction")

# Configuration
config = {
    "pretrained_model": "gpt2-medium",
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 5e-5
}

class TweetsDataset(Dataset):
    def __init__(self, encodings, scores):
        self.encodings = encodings
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item

class TweetPopularityModel(pl.LightningModule):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = GPT2Model.from_pretrained(pretrained_model)
        self.regressor = torch.nn.Linear(self.model.config.n_embd, 1)
        self.validation_losses = []  # To store validation losses for computing metrics later

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.regressor(outputs.last_hidden_state[:, -1, :])
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        outputs = self.forward(**batch)
        loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        outputs = self.forward(**batch)
        val_loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels)
        self.validation_losses.append(val_loss)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss, 'labels': labels, 'predictions': outputs}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_losses).mean()
        self.log('avg_val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        metrics = compute_metrics([x['labels'].cpu().numpy() for x in self.validation_losses],
                                  [x['predictions'].detach().cpu().numpy() for x in self.validation_losses])
        self.log_dict(metrics)
        self.validation_losses = []  # Reset for the next epoch

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=config['learning_rate'])

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained(config['pretrained_model'])
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for padding

    df = pd.read_csv('dataset.csv')
    df['Content'] = df['Content'].apply(preprocess_text)

    df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce')
    df['Analytics'] = pd.to_numeric(df['Analytics'], errors='coerce')
    df.fillna({'Likes': 0, 'Analytics': 1}, inplace=True)

    df['Popularity_Score'] = df['Likes'] / df['Analytics']

    X_train, X_val, y_train, y_val = train_test_split(df['Content'], df['Popularity_Score'], test_size=0.2, random_state=42)

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)

    train_dataset = TweetsDataset(train_encodings, y_train.tolist())
    val_dataset = TweetsDataset(val_encodings, y_val.tolist())

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = TweetPopularityModel(config['pretrained_model'])

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback],
        logger=WandbLogger()
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
