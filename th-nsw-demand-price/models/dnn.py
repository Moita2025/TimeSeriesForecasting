import torch.nn as nn


class DNNForecaster(nn.Module):
    """
    簡單的前饋深度神經網路（MLP），將時間序列扁平化後直接預測
    輸出形狀固定為 (batch, horizon, n_targets)
    """
    def __init__(
        self,
        input_size: int,           # 單一時間步的特徵數
        horizon: int,
        n_targets: int = 2,
        hidden_sizes: list = [256, 128, 64],
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        # 實際輸入維度 = seq_len * input_size（特徵數）
        seq_len = kwargs.get("seq_len", 336)  # 必須從 config 傳入 seq_len
        flatten_dim = seq_len * input_size

        layers = []
        prev_size = flatten_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.dnn_stack = nn.Sequential(*layers)

        # 輸出層 → horizon * n_targets
        self.output_fc = nn.Linear(prev_size, horizon * n_targets)

        # 可選：Softplus 確保輸出非負（視任務而定）
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)          # (batch, seq_len * input_size)

        out = self.dnn_stack(x_flat)                # (batch, last_hidden)
        out = self.output_fc(out)                   # (batch, horizon * n_targets)

        #  reshape 成 (batch, horizon, n_targets)
        out = out.view(batch_size, self.horizon, self.n_targets)

        # 若任務需求輸出非負，可啟用
        # out = self.softplus(out)

        return out