import torch
from torch import nn


class LandmarkLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        return self.classifier(outputs[:, -1])
