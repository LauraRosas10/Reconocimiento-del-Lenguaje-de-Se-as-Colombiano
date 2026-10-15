import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs + self.pe[:, : inputs.size(1)])


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        frame_count: int,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.35,
        pooling: str = "token",
        use_sinusoidal: bool = False,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"token", "mean", "mean_max"}:
            raise ValueError(f"Unknown pooling method: {pooling}")
        self.pooling = pooling
        self.use_sinusoidal = use_sinusoidal
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.class_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.empty(1, frame_count + 1, hidden_size))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if use_sinusoidal:
            self.pos_encoder = PositionalEncoding(
                hidden_size,
                max_len=max(frame_count + 10, 256),
                dropout=dropout * 0.5,
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        classifier_size = hidden_size * 2 if pooling == "mean_max" else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(inputs)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        embedded = torch.cat((class_token, embedded), dim=1)
        if self.use_sinusoidal:
            embedded = self.pos_encoder(embedded)
        else:
            embedded = embedded + self.position_embedding[:, : embedded.shape[1]]
        encoded = self.encoder(embedded)
        if self.pooling == "token":
            pooled = encoded[:, 0]
        elif self.pooling == "mean":
            pooled = encoded[:, 1:].mean(dim=1)
        else:
            pooled = torch.cat(
                (encoded[:, 1:].mean(dim=1), encoded[:, 1:].max(dim=1).values),
                dim=-1,
            )
        return self.classifier(pooled)


class LandmarkBiGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_norm(inputs))
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        return self.classifier(pooled)


class LandmarkConv1DNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout * 0.5)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2)
        self.norm3 = nn.BatchNorm1d(hidden_size * 2)
        self.shortcut = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.act1(self.norm1(self.in_proj(inputs.transpose(1, 2))))
        residual = features
        features = self.drop1(self.act1(self.norm2(self.conv1(features)))) + residual
        residual = self.shortcut(features)
        features = self.drop2(self.act2(self.norm3(self.conv2(features)))) + residual
        pooled = torch.cat((features.mean(dim=-1), features.max(dim=-1).values), dim=-1)
        return self.classifier(pooled)


class LandmarkHybridConvGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        conv_channels: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden * 4),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.gru(features)
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        return self.classifier(pooled)


def create_model(
    model_name: str,
    input_size: int,
    num_classes: int,
    frame_count: int,
    **kwargs,
) -> nn.Module:
    name = model_name.lower().strip()
    if name in {"transformer", "landmark_transformer"}:
        return LandmarkTransformer(input_size, num_classes, frame_count, **kwargs)
    if name in {"bigru", "gru", "landmark_bigru"}:
        return LandmarkBiGRU(input_size, num_classes, **kwargs)
    if name in {"conv1d", "cnn", "landmark_conv1d"}:
        return LandmarkConv1DNet(input_size, num_classes, **kwargs)
    if name in {"hybrid", "conv_gru", "landmark_hybrid"}:
        return LandmarkHybridConvGRU(input_size, num_classes, **kwargs)
    raise ValueError(
        f"Unsupported model_name {model_name!r}. Choose from: "
        "'transformer', 'bigru', 'conv1d', 'hybrid'."
    )
