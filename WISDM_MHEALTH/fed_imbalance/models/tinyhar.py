import torch, torch.nn as nn

class TinyHAR(nn.Module):
    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()
        # temporal convs
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) -> conv expects (B, C, T)
        x = x.transpose(1,2)
        x = self.conv(x)           # (B, 128, T)
        x = x.transpose(1,2)       # (B, T, 128)
        out,_ = self.lstm(x)       # (B, T, 256)
        # attention over time
        a = self.attn(out)         # (B, T, 1)
        w = torch.softmax(a, dim=1)
        feat = (out * w).sum(dim=1)  # (B, 256)
        logits = self.fc(feat)
        return logits
