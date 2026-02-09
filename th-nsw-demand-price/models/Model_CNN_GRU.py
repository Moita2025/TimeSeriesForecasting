import torch.nn as nn

class CNNGRUForecaster(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        # 从 config 读取关键参数，设置合理默认值
        self.input_size     = config.get('input_size')          # 必须提供
        self.hidden_size    = config.get('hidden_size', 128)
        self.num_gru_layers = config.get('num_gru_layers', 2)
        self.num_cnn_filters= config.get('num_cnn_filters', 64)
        self.kernel_size    = config.get('kernel_size', 3)
        self.dropout        = config.get('dropout', 0.2)
        self.output_size    = config.get('output_size', 2)      # 默认改为 2（demand + price）
        self.bidirectional  = config.get('bidirectional', False)
        self.use_pool       = config.get('use_pool', False)

        # CNN 部分
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.num_cnn_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True
        )
        self.relu = nn.ReLU()
        
        if self.use_pool:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # GRU 部分
        self.gru = nn.GRU(
            input_size=self.num_cnn_filters,
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_gru_layers > 1 else 0
        )
        
        # 输出层维度调整
        gru_output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(gru_output_dim, self.output_size)
        self.softplus = nn.Softplus()   # 可选，视任务而定（负荷/价格不一定需要）

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)           # → (batch, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        
        if hasattr(self, 'pool'):
            x = self.pool(x)
        
        x = x.permute(0, 2, 1)           # → (batch, new_seq_len, filters)
        
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]          # 最后时间步
        
        out = self.dropout_layer(out)
        out = self.fc(out)
        out = self.softplus(out)         # 可注释掉，视任务
        
        return out