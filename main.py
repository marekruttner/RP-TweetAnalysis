import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2Tokenizer, AdamW

from utils import preprocess_text  # Adjust the preprocess_text function for minimal preprocessing

import wandb

# Initialize wandb
wandb.init(project="tweet-popularity-prediction")

# Configuration
config = {
    "pretrained_model": "gpt2-medium",  # Example model, adjust based on your needs and resource availability
    "epochs": 20,
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

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = self.regressor(output.last_hidden_state[:, -1, :])
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        outputs = self.forward(**batch)
        loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels)
        self.log('train_loss', loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True)  # Log val_loss only on epoch end
        return loss

    def validation_step(self, batch, batch_idx):
        # Similar to training_step, calculate loss on validation data
        labels = batch.pop('labels')
        outputs = self.forward(**batch)
        loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

class SaveBestModel(Callback):
    def __init__(self, monitor="val_loss", save_top_k=1, mode="min"):
        super().__init__()
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Saves the model with the best validation score."""
        current_score = getattr(trainer.callback_metrics, self.monitor)
        checkpoint = ModelCheckpoint(dirpath=trainer.checkpoint_dirpath,
                                     filename="best_model.ckpt",
                                     monitor=self.monitor,
                                     save_top_k=self.save_top_k,
                                     mode=self.mode)
        checkpoint.on_validation_end(trainer, pl_module)  # Leverage ModelCheckpoint callback


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained(config['pretrained_model'])

    # Set padding token to EOS token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv('dataset.csv')
    df['Content'] = df['Content'].apply(preprocess_text)

    df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce')
    df['Analytics'] = pd.to_numeric(df['Analytics'], errors='coerce')
    df.fillna({'Likes': 0, 'Analytics': 1}, inplace=True)

    df['Popularity_Score'] = df['Likes'] / df['Analytics']

    X_train, X_val, y_train, y_val = train_test_split(df['Content'], df['Popularity_Score'], test_size=0.2,
                                                      random_state=42)

    # Now that the tokenizer has a pad_token, padding should work
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)

    train_dataset = TweetsDataset(train_encodings, y_train.tolist())
    val_dataset = TweetsDataset(val_encodings, y_val.tolist())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = TweetPopularityModel(pretrained_model=config['pretrained_model'])

    trainer = pl.Trainer(max_epochs=config['epochs'], callbacks=[ModelCheckpoint(monitor='val_loss')],
                         logger=WandbLogger(project="tweet-popularity-prediction"))
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
