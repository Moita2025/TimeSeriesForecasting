import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        dim = hidden_size * 2
        self.attn = nn.Linear(dim, dim)           # ← 這層負責 energy
        self.score_proj = nn.Linear(dim, 1, bias=False)  # ← 專門投影到 scalar score

    def forward(self, lstm_out):
        # lstm_out: (B, T, dim)
        energy = torch.tanh(self.attn(lstm_out))          # (B, T, dim)
        scores = self.score_proj(energy).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)            # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, dim)
        return context


class AMBiLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_targets: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = Attention(hidden_size)

        dim = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, horizon * n_targets)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)     # (B, dim)
        out = self.dropout(attn_out)
        out = self.fc(out)
        out = out.view(-1, self.horizon, self.n_targets)
        return out