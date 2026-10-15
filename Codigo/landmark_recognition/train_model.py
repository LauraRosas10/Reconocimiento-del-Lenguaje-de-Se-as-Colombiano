import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from landmark_recognition.data import augment_landmark_sequence, normalize_landmark_sequence
from landmark_recognition.lstm_recurrente.model import LandmarkLSTM
from landmark_recognition.models import (
    LandmarkBiGRU,
    LandmarkConv1DNet,
    LandmarkHybridConvGRU,
    LandmarkTransformer,
)


def _make_splits(
    labels: np.ndarray,
    signers: np.ndarray,
    validation_signer: str,
    test_signer: str,
    seed: int,
    split_mode: str = "subject",
):
    signer_names = sorted(set(signers.tolist()))
    if split_mode == "random":
        if validation_signer or test_signer:
            raise ValueError("Do not specify participant IDs when split_mode='random'.")
        generator = np.random.default_rng(seed)
        train_parts = []
        validation_parts = []
        test_parts = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            generator.shuffle(indices)
            test_count = max(1, int(round(len(indices) * 0.1)))
            validation_count = max(1, int(round(len(indices) * 0.1)))
            if len(indices) - test_count - validation_count < 1:
                raise ValueError(f"Class {label} has too few samples for stratified train/validation/test splits.")
            test_parts.append(indices[:test_count])
            validation_parts.append(indices[test_count : test_count + validation_count])
            train_parts.append(indices[test_count + validation_count :])
        return (
            np.concatenate(train_parts),
            np.concatenate(validation_parts),
            np.concatenate(test_parts),
        )
    if split_mode != "subject":
        raise ValueError(f"Unsupported split mode: {split_mode!r}")

    if not validation_signer and len(signer_names) >= 3:
        validation_signer = signer_names[-2]
    if not test_signer and len(signer_names) >= 3:
        test_signer = signer_names[-1]

    if validation_signer or test_signer:
        if validation_signer and validation_signer not in signer_names:
            raise ValueError(f"Unknown validation signer {validation_signer!r}; available: {signer_names}")
        if test_signer and test_signer not in signer_names:
            raise ValueError(f"Unknown test signer {test_signer!r}; available: {signer_names}")
        if validation_signer and validation_signer == test_signer:
            raise ValueError("Validation and test signers must be different.")
        val_indices = np.flatnonzero(signers == validation_signer) if validation_signer else np.array([], dtype=int)
        test_indices = np.flatnonzero(signers == test_signer) if test_signer else np.array([], dtype=int)
        excluded = set(val_indices.tolist()) | set(test_indices.tolist())
        train_indices = np.asarray([index for index in range(len(signers)) if index not in excluded])
        if not len(train_indices) or not len(val_indices):
            raise ValueError("The selected signers must leave at least one training and one validation sample.")
        return train_indices, val_indices, test_indices

    if len(signers) < 2:
        raise ValueError("At least 2 sequences are required for a train/validation split.")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(signers))
    test_count = max(1, len(indices) // 10)
    validation_count = max(1, len(indices) // 10)
    return indices[test_count + validation_count :], indices[test_count : test_count + validation_count], indices[:test_count]


def _build_model(
    architecture: str,
    input_size: int,
    num_classes: int,
    frame_count: int,
    pooling: str = "mean",
    dropout: float = 0.35,
):
    if architecture == "lstm":
        return LandmarkLSTM(input_size=input_size, num_classes=num_classes)
    elif architecture == "bigru":
        return LandmarkBiGRU(input_size=input_size, num_classes=num_classes, hidden_size=64, layers=2, dropout=dropout)
    elif architecture == "conv1d":
        return LandmarkConv1DNet(input_size=input_size, num_classes=num_classes, hidden_size=64, dropout=dropout)
    elif architecture == "hybrid":
        return LandmarkHybridConvGRU(input_size=input_size, num_classes=num_classes, conv_channels=64, gru_hidden=64, dropout=dropout)
    return LandmarkTransformer(
        input_size=input_size,
        num_classes=num_classes,
        frame_count=frame_count,
        hidden_size=64,
        heads=4,
        layers=1,
        dropout=dropout,
        pooling=pooling,
        norm_first=True,
    )


def _evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            total_loss += criterion(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions.cpu() == labels.cpu()).sum())
            total += len(labels)
    return (total_loss / total, correct / total) if total else (float("nan"), float("nan"))


def _predictions(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, labels in loader:
            predictions.extend(model(features.to(device)).argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return predictions, targets


class LandmarkTrainingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        augment: bool,
        seed: int,
        include_velocity: bool = False,
    ) -> None:
        self.features = features
        self.labels = labels
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.include_velocity = include_velocity

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        sequence = self.features[index]
        if self.augment:
            sequence = augment_landmark_sequence(sequence, rng=self.rng)
        sequence = sequence.reshape(sequence.shape[0], -1)
        sequence = ((sequence - self.feature_mean) / self.feature_std).astype(np.float32)
        if self.include_velocity:
            vel = np.zeros_like(sequence)
            vel[1:] = sequence[1:] - sequence[:-1]
            sequence = np.concatenate([sequence, vel], axis=-1).astype(np.float32)
        return torch.from_numpy(sequence), torch.tensor(self.labels[index], dtype=torch.long)


def _prepare_eval_tensor(
    flattened_data: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    include_velocity: bool,
) -> torch.Tensor:
    normed = ((flattened_data - feature_mean) / feature_std).astype(np.float32)
    if not include_velocity:
        return torch.from_numpy(normed)
    vel = np.zeros_like(normed)
    vel[:, 1:] = normed[:, 1:] - normed[:, :-1]
    return torch.from_numpy(np.concatenate([normed, vel], axis=-1).astype(np.float32))


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.dataset, allow_pickle=False) as data:
        features = data["X"].astype(np.float32)
        labels = data["y"].astype(np.int64)
        signers = data["signers"].astype(str)
        class_names = data["class_names"].astype(str).tolist()
    if features.ndim != 4 or features.shape[0] != len(labels) or len(signers) != len(labels):
        raise ValueError("Dataset must contain aligned X [N,T,landmarks,3], y, and signers arrays.")

    frame_count = features.shape[1]

    wrist_relative = args.wrist_relative_hands or args.signer_invariant
    include_velocity = args.include_velocity or args.signer_invariant
    no_mean = args.no_mean_shift or args.signer_invariant

    if args.signer_invariant:
        print("Signer-invariant mode enabled: wrist-relative hand shapes + velocities + zero-centered normalization.")

    features = np.stack([
        normalize_landmark_sequence(sequence, wrist_relative_hands=wrist_relative)
        for sequence in features
    ])
    flattened = features.reshape(features.shape[0], frame_count, -1)
    train_indices, validation_indices, test_indices = _make_splits(
        labels,
        signers,
        args.validation_signer,
        args.test_signer,
        args.seed,
        args.split_mode,
    )
    print(
        f"Split mode={args.split_mode}; "
        f"train={len(train_indices)}, validation={len(validation_indices)}, "
        f"test={len(test_indices)}"
    )
    if args.split_mode == "subject":
        print(
            "Participants: "
            f"train={sorted(set(signers[train_indices].tolist()))}, "
            f"validation={sorted(set(signers[validation_indices].tolist()))}, "
            f"test={sorted(set(signers[test_indices].tolist()))}"
        )
    else:
        print(
            "Warning: random split includes participants in multiple subsets; "
            "use it only as a learnability diagnostic, not as unseen-signer performance."
        )
    for subset_name, subset_indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        if len(subset_indices):
            class_counts = np.bincount(labels[subset_indices], minlength=len(class_names))
            absent = np.flatnonzero(class_counts == 0)
            print(
                f"{subset_name} class counts: min={int(class_counts.min())}, "
                f"max={int(class_counts.max())}, missing={absent.tolist()}"
            )

    if no_mean:
        feature_mean = np.zeros(flattened.shape[-1], dtype=np.float32)
    else:
        feature_mean = flattened[train_indices].mean(axis=(0, 1), keepdims=True).reshape(-1)
    feature_std = flattened[train_indices].std(axis=(0, 1), keepdims=True).reshape(-1)
    feature_std = np.maximum(feature_std, 1e-4)

    input_features_dim = flattened.shape[-1] * (2 if include_velocity else 1)

    augment = not args.no_augmentation
    print(f"Training augmentation: {'enabled' if augment else 'disabled'} (online; labels unchanged)")
    train_dataset = LandmarkTrainingDataset(
        features[train_indices],
        labels[train_indices],
        feature_mean,
        feature_std,
        augment=augment,
        seed=args.seed,
        include_velocity=include_velocity,
    )
    train_eval_loader = DataLoader(
        TensorDataset(
            _prepare_eval_tensor(flattened[train_indices], feature_mean, feature_std, include_velocity),
            torch.from_numpy(labels[train_indices]),
        ),
        batch_size=args.batch_size,
    )
    validation_loader = DataLoader(
        TensorDataset(
            _prepare_eval_tensor(flattened[validation_indices], feature_mean, feature_std, include_velocity),
            torch.from_numpy(labels[validation_indices]),
        ),
        batch_size=args.batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = (
        DataLoader(
            TensorDataset(
                _prepare_eval_tensor(flattened[test_indices], feature_mean, feature_std, include_velocity),
                torch.from_numpy(labels[test_indices]),
            ),
            batch_size=args.batch_size,
        )
        if len(test_indices)
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.4 if (args.signer_invariant or args.split_mode == "subject") else 0.35
    weight_decay = 0.01 if (args.signer_invariant or args.split_mode == "subject") else 1e-3
    pooling = args.pooling

    model = _build_model(
        args.architecture,
        input_features_dim,
        len(class_names),
        frame_count,
        pooling=pooling,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    early_stopping_patience = args.early_stopping_patience
    best_validation_loss = float("inf")
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)

        train_loss = total_loss / len(train_indices)
        train_accuracy = _evaluate(model, train_eval_loader, criterion, device)[1]
        validation_loss, validation_accuracy = _evaluate(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, "
            f"val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2g}"
        )
        if validation_loss < best_validation_loss - args.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_validation_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}; best validation loss "
                    f"{best_validation_loss:.4f} at epoch {best_epoch}."
                )
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")
    model.load_state_dict(best_state)
    test_accuracy = _evaluate(model, test_loader, criterion, device)[1] if test_loader else None
    test_predictions, test_targets = (
        _predictions(model, test_loader, device) if test_loader else ([], [])
    )
    checkpoint = {
        "architecture": args.architecture,
        "input_size": int(input_features_dim),
        "frame_count": int(frame_count),
        "num_classes": len(class_names),
        "class_names": class_names,
        "signer_invariant": bool(args.signer_invariant),
        "wrist_relative_hands": bool(wrist_relative),
        "include_velocity": bool(include_velocity),
        "pooling": pooling,
        "norm_first": True,
        "dropout": dropout,
        "heads": 4,
        "state_dict": best_state,
        "validation_accuracy": best_validation_accuracy,
        "validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "early_stopping_patience": early_stopping_patience,
        "test_accuracy": test_accuracy,
        "history": history,
        "test_predictions": test_predictions,
        "test_targets": test_targets,
        "feature_mean": torch.from_numpy(feature_mean.reshape(-1)),
        "feature_std": torch.from_numpy(feature_std.reshape(-1)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    if test_accuracy is not None:
        print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved checkpoint to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer, GRU, or LSTM on prepared landmark sequences.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("transformer", "lstm", "bigru", "conv1d", "hybrid"),
        default="transformer",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/transformer.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-signer", default="")
    parser.add_argument("--test-signer", default="")
    parser.add_argument("--split-mode", choices=("subject", "random"), default="subject")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable online landmark augmentation.")
    parser.add_argument(
        "--signer-invariant",
        action="store_true",
        help="Enable full signer-invariant representation (wrist-relative hands + velocities + zero-centered) to test unseen persons.",
    )
    parser.add_argument(
        "--wrist-relative-hands",
        action="store_true",
        help="Center finger coordinates on their respective wrist.",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Append temporal velocity features (delta_x) to input.",
    )
    parser.add_argument(
        "--no-mean-shift",
        action="store_true",
        help="Do not subtract train mean from features (preserves true zero coordinates for unseen signers).",
    )
    parser.add_argument(
        "--pooling",
        choices=("token", "mean", "mean_max"),
        default="mean",
        help="Temporal pooling method for LandmarkTransformer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.early_stopping_patience < 1
        or args.early_stopping_min_delta < 0
    ):
        parser.error(
            "epochs, batch-size, and early-stopping-patience must be positive; "
            "learning-rate and early-stopping-min-delta must be non-negative."
        )
    if args.split_mode == "random" and (args.validation_signer or args.test_signer):
        parser.error("--validation-signer and --test-signer cannot be used with --split-mode random.")
    train(args)


if __name__ == "__main__":
    main()
