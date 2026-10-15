"""Training and evaluation routines optimized for scarce sign-language landmark data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import augment_landmark_sequence, prepare_features
from .models import create_model


class LandmarkDataset(Dataset):
    """PyTorch Dataset for sign language landmark sequences.
    
    Supports:
    - On-the-fly data augmentation during training.
    - Automatic inclusion of first-order velocity features (positions + velocities).
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        augment: bool = False,
        include_velocity: bool = True,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.include_velocity = include_velocity

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[index]  # Expected shape: [frames, LANDMARK_COUNT, 3]

        if self.augment:
            seq = augment_landmark_sequence(seq)

        # Compute flat features + velocities: [frames, 225] or [frames, 450]
        feats = prepare_features(seq, include_velocity=self.include_velocity)
        return torch.from_numpy(feats), torch.tensor(self.labels[index], dtype=torch.long)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains the model for one epoch and returns the average training loss."""
    model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model and returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=-1)

        total_loss += loss.item() * len(labels)
        correct += (preds == labels).sum().item()

    dataset_size = len(dataloader.dataset)
    return total_loss / dataset_size, correct / dataset_size


def train_model(
    model_name: str,
    train_sequences: List[np.ndarray],
    train_labels: List[int],
    val_sequences: List[np.ndarray],
    val_labels: List[int],
    num_classes: int,
    frame_count: int,
    include_velocity: bool = True,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 15,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Trains a model with early stopping, label smoothing, and cosine decay lr scheduler.
    
    Args:
        model_name: 'bigru', 'conv1d', 'hybrid', or 'transformer'.
        train_sequences: List of arrays shaped [frames, 75, 3].
        train_labels: List of integer class indices.
        val_sequences: Validation sequence arrays.
        val_labels: Validation integer labels.
        num_classes: Total number of classes.
        frame_count: Number of frames per sample.
        include_velocity: Whether to append first-order velocities (recommended).
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        weight_decay: L2 regularization penalty.
        label_smoothing: Smoothing factor for CrossEntropyLoss (prevents overfitting).
        patience: Early stopping patience epochs.
        device: Torch device (defaults to CUDA if available, else CPU).
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LandmarkDataset(train_sequences, train_labels, augment=True, include_velocity=include_velocity)
    val_ds = LandmarkDataset(val_sequences, val_labels, augment=False, include_velocity=include_velocity)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 75 landmarks * 3 coordinates = 225; with velocity = 450
    input_size = 75 * 3 * (2 if include_velocity else 1)

    model = create_model(
        model_name=model_name,
        input_size=input_size,
        num_classes=num_classes,
        frame_count=frame_count,
    ).to(dev)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, dev)
        val_loss, val_acc = evaluate(model, val_loader, criterion, dev)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history

