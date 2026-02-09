import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils_Non_Tree import *  # 复用所有通用函数：load_daily_data, prepare_data, create_sequences, TimeSeriesDataset, train_model, rolling_forecast_and_evaluate

# ────────────────────────────────────────────────
# 设备选择（与 BiLSTM 一致）
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# DNN 模型（输入扁平化处理，Softplus 限幅输出）
# ────────────────────────────────────────────────
class DNNForecaster(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.2, output_size=1):
        super().__init__()
        # 输入扁平化：seq_length * num_features → 第一隐藏层
        self.flatten_input_size = input_size  # input_size = seq_length * num_features（create_sequences 后自动处理）
        
        layers = []
        prev_size = self.flatten_input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # 加速收敛 + 稳定
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.dnn_stack = nn.Sequential(*layers)
        self.output_fc = nn.Linear(prev_size, output_size)
        self.softplus = nn.Softplus()  # 限幅确保正值（响应师兄建议，与 BiLSTM 一致）

    def forward(self, x):
        # x: (batch, seq_length, num_features) → (batch, seq_length * num_features)
        batch_size = x.size(0)
        seq_len, feat_dim = x.size(1), x.size(2)
        x_flat = x.reshape(batch_size, -1)  # 扁平化
        
        out = self.dnn_stack(x_flat)
        out = self.output_fc(out)
        out = self.softplus(out)  # 输出 > 0，物理含义（发电量）
        return out

# ────────────────────────────────────────────────
# 主程式（与 BiLSTM 完全一致，便于对比）
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 14          # DNN 也用序列输入（扁平化处理），保持一致
    PRED_LENGTH   = 1           # 每日尺度
    TEST_RATIO    = 0.2
    LR            = 0.001
    EPOCHS        = 100
    PATIENCE      = 15
    BATCH_SIZE    = 64

    # 载入与准备（复用 Utils）
    df = load_daily_data()
    (X_train, X_test, y_train, y_test, y_test_orig,
     df_test, scaler_X, scaler_y, feature_cols) = prepare_data(df, LAG_WINDOW, TEST_RATIO)

    print("Features:", feature_cols)
    print(f"Input size for DNN: SEQ_LENGTH({SEQ_LENGTH}) * num_features({len(feature_cols)}) = {SEQ_LENGTH * len(feature_cols)}")

    # 建立序列（复用 Utils，完全一致）
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_ts_seq, y_ts_seq = create_sequences(X_test,  y_test,  SEQ_LENGTH, PRED_LENGTH)

    # 分出驗證集（复用 BiLSTM 逻辑）
    val_ratio = 0.15
    tr_idx = int(len(X_tr_seq) * (1 - val_ratio))
    X_tr, X_val = X_tr_seq[:tr_idx], X_tr_seq[tr_idx:]
    y_tr, y_val = y_tr_seq[:tr_idx], y_tr_seq[tr_idx:]

    # DataLoader（复用）
    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型初始化（注意：input_size = seq_length * num_features）
    model = DNNForecaster(
        input_size=SEQ_LENGTH * len(feature_cols),  # 扁平化输入尺寸
        hidden_sizes=[256, 128, 64],
        dropout=0.2,
        output_size=PRED_LENGTH
    ).to(device)

    print(model)

    # 訓練（复用 Utils）
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device)

    # 滾動預測評估（复用 Utils，完全一致 pipeline）
    rolling_forecast_and_evaluate(
        model, df_test, feature_cols, scaler_y,
        seq_length=SEQ_LENGTH, forecast_horizon=PRED_LENGTH, device=device
    )