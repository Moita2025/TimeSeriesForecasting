import torch.nn as nn


class BiLSTMForecaster(nn.Module):
    """
    雙向 LSTM 預測器，適用於多步時間序列預測
    輸出形狀固定為 (batch, horizon, n_targets)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        horizon: int = 48,
        n_targets: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, horizon * n_targets)

        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)                      # (batch, seq_len, out_dim)
        out = lstm_out[:, -1, :]                        # 取最後一個時間步 (batch, out_dim)
        out = self.dropout(out)
        out = self.fc(out)                              # (batch, horizon * n_targets)
        out = out.view(-1, self.horizon, self.n_targets)  # (batch, horizon, n_targets)
        return out