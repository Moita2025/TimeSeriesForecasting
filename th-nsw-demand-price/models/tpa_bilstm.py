import torch
import torch.nn as nn


class TemporalPatternAttention(nn.Module):
    """
    Temporal Pattern Attention (TPA) 模塊
    參考：Temporal Pattern Attention for Multivariate Time Series Forecasting (2019)
    """
    def __init__(self, hidden_size: int, time_step: int):
        super().__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size

        dim = hidden_size * 2  # 因為 bidirectional

        self.w_h = nn.Linear(dim, dim, bias=False)
        self.w_s = nn.Linear(dim, dim, bias=False)
        self.v   = nn.Parameter(torch.randn(dim))
        self.w_c = nn.Linear(dim, dim)

    def forward(self, lstm_out):
        # lstm_out: (B, T, D)  D=hidden*2
        h = self.w_h(lstm_out)                                 # (B,T,D)
        s = self.w_s(lstm_out[:, -1:, :].expand(-1, self.time_step, -1))  # (B,T,D)

        e = torch.sum(self.v * torch.tanh(h + s), dim=-1)      # (B,T)
        alpha = torch.softmax(e, dim=-1)                       # (B,T)

        c = torch.bmm(alpha.unsqueeze(1), lstm_out).squeeze(1) # (B,D)
        c = torch.tanh(self.w_c(c))
        return c, alpha


class TPABiLSTMForecaster(nn.Module):
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

        self.tpa = TemporalPatternAttention(
            hidden_size=hidden_size,
            time_step=kwargs.get("seq_len", 336)   # 務必從 config 傳入 seq_len
        )

        dim = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, horizon * n_targets)

    def forward(self, x):
        # x: (B, seq_len, input_size)
        lstm_out, _ = self.lstm(x)              # (B, seq_len, hidden*2)

        context, _ = self.tpa(lstm_out)         # (B, hidden*2)

        out = self.dropout(context)
        out = self.fc(out)                      # (B, horizon * n_targets)
        out = out.view(-1, self.horizon, self.n_targets)
        return out