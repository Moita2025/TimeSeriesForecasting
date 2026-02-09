import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils_Non_Tree import *

# ────────────────────────────────────────────────
# 設備選擇
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CNN-GRU 模型
# ────────────────────────────────────────────────
class CNNGRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_gru_layers=1, 
                 num_cnn_filters=64, kernel_size=3, dropout=0.2, output_size=1):
        super().__init__()
        
        # CNN 部分：提取局部特徵
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2,   # 保持序列長度不變
            bias=True
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 可選：降採樣
        
        # 計算經過 CNN + pool 後的序列長度變化（這裡先不 pool 也可）
        # 如果有 pool，需計算新的 seq_len，但這裡暫時保持原長度
        
        # GRU 部分
        self.gru = nn.GRU(
            input_size=num_cnn_filters,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=False,         # 可改成 True 變成 BiGRU
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層
        # 如果有 pool，需要調整 *2；這裡假設不 pool，維持原序列長度
        self.fc = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # Conv1d 需要 (batch, channels=features, seq_len)
        x = x.permute(0, 2, 1)              # → (batch, input_size, seq_len)
        x = self.conv1(x)                    # → (batch, num_filters, seq_len)
        x = self.relu(x)
        # x = self.pool(x)                   # 可選：若加入則需後續調整維度
        
        # 轉回 (batch, seq_len', num_filters) 給 GRU
        x = x.permute(0, 2, 1)               # → (batch, seq_len, num_filters)
        
        gru_out, _ = self.gru(x)             # (batch, seq_len, hidden_size)
        out = gru_out[:, -1, :]              # 取最後時間步
        
        out = self.dropout(out)
        out = self.fc(out)                   # (batch, output_size)
        out = self.softplus(out)
        
        return out


# ────────────────────────────────────────────────
# 主程式
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 14          # 可自行調整 7~30
    PRED_LENGTH   = 1           # 每日尺度先預測 1 天
    TEST_RATIO    = 0.2
    HIDDEN_SIZE   = 128
    DROPOUT       = 0.2
    LR            = 0.001
    EPOCHS        = 100
    PATIENCE      = 15
    BATCH_SIZE    = 64

    # CNN-GRU 專屬參數（可調整）
    NUM_CNN_FILTERS = 64
    KERNEL_SIZE     = 3

    # 載入與準備
    df = load_daily_data()
    (X_train, X_test, y_train, y_test, y_test_orig,
     df_test, scaler_X, scaler_y, feature_cols) = prepare_data(df, LAG_WINDOW, TEST_RATIO)

    print("Features:", feature_cols)

    # 建立序列
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_ts_seq, y_ts_seq = create_sequences(X_test,  y_test,  SEQ_LENGTH, PRED_LENGTH)

    # 分出驗證集
    val_ratio = 0.15
    tr_idx = int(len(X_tr_seq) * (1 - val_ratio))
    X_tr, X_val = X_tr_seq[:tr_idx], X_tr_seq[tr_idx:]
    y_tr, y_val = y_tr_seq[:tr_idx], y_tr_seq[tr_idx:]

    # DataLoader
    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = CNNGRUForecaster(
        input_size       = len(feature_cols),
        hidden_size      = HIDDEN_SIZE,
        num_gru_layers   = 2,
        num_cnn_filters  = NUM_CNN_FILTERS,
        kernel_size      = KERNEL_SIZE,
        dropout          = DROPOUT,
        output_size      = PRED_LENGTH
    ).to(device)

    print(model)

    # 訓練
    model = train_model(model, train_loader, val_loader, 
                        epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device)

    # 滾動預測評估
    rolling_forecast_and_evaluate(
        model, df_test, feature_cols, scaler_y,
        seq_length=SEQ_LENGTH, forecast_horizon=PRED_LENGTH, device=device
    )