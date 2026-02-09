import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils_Non_Tree import *

# ────────────────────────────────────────────────
# 設備選擇 → 相同，省略
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# 4. TPA-BiLSTM 模型（核心改動）
# ────────────────────────────────────────────────
class TemporalPatternAttention(nn.Module):
    """
    Temporal Pattern Attention (TPA) 模塊
    參考論文：Temporal Pattern Attention for Multivariate Time Series Forecasting (2019)
    """
    def __init__(self, input_dim, hidden_size, time_step):
        super().__init__()
        self.time_step = time_step          # seq_length
        self.hidden_size = hidden_size

        # 用來計算每個時間步的「模式相似度」
        self.w_h = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)   # 雙向 BiLSTM
        self.w_s = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_size * 2))

        # 最終的加權特徵投影
        self.w_c = nn.Linear(hidden_size * 2, hidden_size * 2)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)

        # 1. 對每個時間步做線性變換
        h = self.w_h(lstm_out)                     # (B, T, D)
        s = self.w_s(lstm_out[:, -1, :].unsqueeze(1).repeat(1, self.time_step, 1))  # 用最後隱狀態作為 query

        # 2. 計算注意力分數 e_{t,i}
        e = torch.sum(self.v * torch.tanh(h + s), dim=-1)   # (B, T)

        # 3. softmax 得到權重
        alpha = torch.softmax(e, dim=-1)                    # (B, T)

        # 4. 加權求和 → context vector
        c = torch.bmm(alpha.unsqueeze(1), lstm_out).squeeze(1)  # (B, hidden*2)

        # 5. 可選：再做一次線性變換
        c = torch.tanh(self.w_c(c))

        return c, alpha   # 返回 context 與 可視化用的 alpha


class TPABiLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2, seq_length=14, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.tpa = TemporalPatternAttention(
            input_dim=input_size,
            hidden_size=hidden_size,
            time_step=seq_length
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)                  # (batch, seq, hidden*2)

        # TPA 注意力
        context, attn_weights = self.tpa(lstm_out)   # context: (batch, hidden*2)

        out = self.dropout(context)
        out = self.fc(out)                           # (batch, output_size)
        out = self.softplus(out)

        return out

# ────────────────────────────────────────────────
# 主程式（改模型類別 + 傳入 seq_length）
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 14          # 重要：TPA 需要知道序列長度
    PRED_LENGTH   = 1
    TEST_RATIO    = 0.2
    HIDDEN_SIZE   = 128
    DROPOUT       = 0.2
    LR            = 0.001
    EPOCHS        = 100
    PATIENCE      = 15
    BATCH_SIZE    = 64

    # 載入與準備（相同）
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

    # 模型 → 改用 TPA-BiLSTM，並傳入 seq_length
    model = TPABiLSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        dropout=DROPOUT,
        seq_length=SEQ_LENGTH,           # ← 新增
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