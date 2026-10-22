import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock1D(12, 32, kernel_size=7),
            ConvBlock1D(32, 64, kernel_size=5),
            ConvBlock1D(64, 128, kernel_size=5),
            ConvBlock1D(128, 128, kernel_size=3)
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(
                "Expected input shape (batch, 12, sequence_length)"
            )

        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)

        return logits
