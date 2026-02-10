import torch.nn as nn

class CNNGRUForecaster(nn.Module):
    """
    CNN + GRU 的混合模型，用於多步時序預測
    輸出形狀固定為 (batch, horizon, n_targets)，這裡 n_targets=2 (demand + price)
    """

    def __init__(
        self,
        input_size: int,           # 特徵數，從 datamodule 傳入
        hidden_size: int = 128,
        num_gru_layers: int = 2,
        num_cnn_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_pool: bool = False,
        horizon: int = 48,
        n_targets: int = 2,        # 預測目標數（demand + price）
    ):
        super().__init__()

        self.horizon = horizon
        self.n_targets = n_targets
        self.output_size = horizon * n_targets

        # CNN 部分
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )
        self.relu = nn.ReLU()

        if use_pool:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # GRU 部分
        gru_input_size = num_cnn_filters
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )

        # 輸出層
        gru_output_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, self.output_size)

        # 可選：最後一層激活（目前不強制 sigmoid / softplus，因為價格可負）
        # self.final_activation = nn.Sigmoid()   # 如果全部要 [0,1] 可開啟

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # CNN 需要 (batch, channels, seq_len)
        x = x.permute(0, 2, 1)          # → (batch, input_size, seq_len)
        x = self.conv1(x)
        x = self.relu(x)

        if hasattr(self, 'pool'):
            x = self.pool(x)

        x = x.permute(0, 2, 1)          # → (batch, reduced_seq_len, num_filters)

        # GRU
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]         # 取最後時間步

        out = self.dropout_layer(out)
        out = self.fc(out)              # (batch, horizon * n_targets)

        # reshape 成 (batch, horizon, n_targets)
        out = out.view(-1, self.horizon, self.n_targets)

        # 可選激活
        # out = self.final_activation(out)

        return out