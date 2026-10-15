from pathlib import Path

import numpy as np
import torch
from torch import nn

from landmark_recognition.data import normalize_landmark_sequence
from landmark_recognition.lstm_recurrente.model import LandmarkLSTM
from landmark_recognition.models import (
    LandmarkBiGRU,
    LandmarkConv1DNet,
    LandmarkHybridConvGRU,
    LandmarkTransformer,
)


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    architecture = checkpoint["architecture"]
    input_size = int(checkpoint["input_size"])
    num_classes = int(checkpoint["num_classes"])
    state_dict = checkpoint["state_dict"]

    if architecture == "lstm":
        recurrent_weight = state_dict["recurrent.weight_hh_l0"]
        hidden_size = recurrent_weight.shape[1]
        layers = 1 + max(
            int(key.split("weight_ih_l", 1)[1].split("_", 1)[0])
            for key in state_dict
            if key.startswith("recurrent.weight_ih_l")
        )
        return LandmarkLSTM(input_size, num_classes, hidden_size=hidden_size, layers=layers)

    if architecture == "bigru":
        hidden_size = state_dict["gru.weight_hh_l0"].shape[0] // 3
        layers = 1 + max(
            int(key.split("weight_ih_l", 1)[1].split("_", 1)[0])
            for key in state_dict
            if key.startswith("gru.weight_ih_l")
        )
        return LandmarkBiGRU(
            input_size,
            num_classes,
            hidden_size=hidden_size,
            layers=layers,
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if architecture == "conv1d":
        hidden_size = state_dict["in_proj.weight"].shape[0]
        return LandmarkConv1DNet(
            input_size,
            num_classes,
            hidden_size=hidden_size,
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if architecture == "hybrid":
        conv_channels = state_dict["feature_extractor.0.weight"].shape[0]
        gru_hidden = state_dict["gru.weight_hh_l0"].shape[0] // 3
        return LandmarkHybridConvGRU(
            input_size,
            num_classes,
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if architecture == "transformer":
        layer_indices = {
            int(key.split("encoder.layers.", 1)[1].split(".", 1)[0])
            for key in state_dict
            if key.startswith("encoder.layers.")
        }
        signer_invariant = bool(checkpoint.get("signer_invariant", False))
        return LandmarkTransformer(
            input_size,
            num_classes,
            int(checkpoint["frame_count"]),
            hidden_size=state_dict["input_projection.weight"].shape[0],
            heads=int(checkpoint.get("heads", 4)),
            layers=max(layer_indices, default=0) + 1,
            dropout=float(checkpoint.get("dropout", 0.35)),
            pooling=checkpoint.get("pooling", "mean" if signer_invariant else "token"),
            use_sinusoidal=bool(checkpoint.get("use_sinusoidal", False)),
            norm_first=bool(checkpoint.get("norm_first", signer_invariant)),
        )

    raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def prepare_checkpoint_features(sequence: np.ndarray, checkpoint: dict) -> np.ndarray:
    normalized = normalize_landmark_sequence(
        sequence,
        wrist_relative_hands=bool(checkpoint.get("wrist_relative_hands", False)),
    )
    features = normalized.reshape(normalized.shape[0], -1)
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    if feature_mean is not None and feature_std is not None:
        features = (
            (features - feature_mean.cpu().numpy()) / feature_std.cpu().numpy()
        ).astype(np.float32)

    if checkpoint.get("include_velocity", False):
        velocity = np.zeros_like(features)
        velocity[1:] = features[1:] - features[:-1]
        features = np.concatenate((features, velocity), axis=-1)

    expected_size = int(checkpoint["input_size"])
    if features.shape[-1] != expected_size:
        raise ValueError(
            f"Checkpoint expects {expected_size} features per frame, "
            f"but preprocessing produced {features.shape[-1]}."
        )
    return features.astype(np.float32, copy=False)
