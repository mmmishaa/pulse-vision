from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

class Trainer:

    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        save_dir: Path,
        model_name: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name

        self.train_loss_history = []
        self.val_loss_history = []

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def validate_one_epoch(self):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for frames, target_ppg in self.val_loader:
                frames, target_ppg = frames.to(self.device), target_ppg.to(self.device)

                pred = self.model(frames)
                loss = self.criterion(pred, target_ppg)
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(self.val_loader)
        self.val_loss_history.append(epoch_loss)

        return epoch_loss

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for frames, target_ppg in self.train_loader:
            frames, target_ppg = frames.to(self.device), target_ppg.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(frames)

            loss = self.criterion(pred, target_ppg)
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(self.train_loader)
        self.train_loss_history.append(epoch_loss)

        return epoch_loss

    def fit(self, epochs: int):
        best_weights_path = self.save_dir / f"{self.model_name}.pt"
        best_val_loss = float("inf")
        best_epoch = 0
        for epoch in tqdm(range(1, epochs + 1), desc=f"Training"):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save(self.model.state_dict(), best_weights_path)

        print(f"Training complete. Weights from epoch {best_epoch} loaded. Val Loss: {best_val_loss:.5f}")
        self.model.load_state_dict(torch.load(best_weights_path, map_location=self.device))