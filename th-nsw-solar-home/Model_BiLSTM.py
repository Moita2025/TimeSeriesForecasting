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
# 4. BiLSTM 模型（調整輸出維度）
# ────────────────────────────────────────────────
class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)           # (batch, seq, 2*hidden)
        out = lstm_out[:, -1, :]             # 取最後一步
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
    model = BiLSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        dropout=DROPOUT,
        output_size=PRED_LENGTH
    ).to(device)

    print(model)

    # 訓練
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device)

    # 滾動預測評估
    rolling_forecast_and_evaluate(
        model, df_test, feature_cols, scaler_y,
        seq_length=SEQ_LENGTH, forecast_horizon=PRED_LENGTH, device=device
    )