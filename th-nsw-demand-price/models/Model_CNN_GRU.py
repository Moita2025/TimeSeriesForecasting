import torch.nn as nn
import torch

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
        self.bidirectional  = config.get('bidirectional', False)
        self.use_pool       = config.get('use_pool', False)

        # 输出维度改为 48 × 2
        self.horizon        = config.get('horizon', 48)
        self.output_size    = 2 * self.horizon   # 96

        # 新增：记录训练集的目标范围（后续在 prepare_data 传进来，或在这里硬编码）
        # 更好的做法是在 config 里传入 min/max
        self.demand_min = config.get('demand_min', 3000.0)   # 后续从训练集取
        self.demand_max = config.get('demand_max', 11000.0)
        self.price_min  = config.get('price_min', -200.0)
        self.price_max  = config.get('price_max',  1500.0)

        # self.fc = nn.Linear(gru_output_dim, self.output_size)

        # 改用 Sigmoid 压缩到 [0,1]
        self.sigmoid = nn.Sigmoid()

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
        
        # 改成输出完整的 horizon × targets
        self.fc = nn.Linear(gru_output_dim, self.output_size)
        
        # 暂时保留 softplus，但强烈建议后续移除（价格可以负）
        # self.softplus = nn.Softplus()

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
        out = self.fc(out)                      # (batch, 96)
        # out = self.sigmoid(out)                 # → [0,1]

        # reshape & 映射到物理范围
        out = out.view(-1, self.horizon, 2)
        return out