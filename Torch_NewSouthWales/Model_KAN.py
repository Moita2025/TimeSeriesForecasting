import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils_Non_Tree import *
from kan import KAN   # ← pip install pykan 后即可使用

# ────────────────────────────────────────────────
# 设备
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# KAN 模型封装（兼容你的 Utils）
# ────────────────────────────────────────────────
class KANForecaster(nn.Module):
    def __init__(self, input_size, hidden_dim=16, output_size=1, grid=5, k=3):
        super().__init__()
        self.kan = KAN(
            width=[input_size, hidden_dim, output_size],   # [输入, 隐藏, 输出]
            grid=grid,
            k=k,
            seed=0,
            device=device
        )
        self.softplus = nn.Softplus()   # 保证预测值非负（能耗）

    def forward(self, x):
        # x shape: (batch, seq_length=1, input_size) → 压成 (batch, input_size)
        x = x.squeeze(1)
        out = self.kan(x)
        return self.softplus(out)


# ────────────────────────────────────────────────
# 主程序（几乎和 Model_BiLSTM.py 一模一样）
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 1          # ← 关键！KAN 是前馈，用 seq=1 即可
    PRED_LENGTH   = 1
    TEST_RATIO    = 0.2
    HIDDEN_DIM    = 32         # 可根据特征数调整（太大容易过拟合）
    GRID          = 5
    K             = 3
    LR            = 0.001
    EPOCHS        = 300        # KAN 通常收敛更快，可适当减小
    PATIENCE      = 20
    BATCH_SIZE    = 64

    # 数据准备（完全复用）
    df = load_daily_data()
    (X_train, X_test, y_train, y_test, y_test_orig,
     df_test, scaler_X, scaler_y, feature_cols) = prepare_data(df, LAG_WINDOW, TEST_RATIO)

    print("Features:", feature_cols)
    print(f"Input dimension for KAN: {len(feature_cols)}")

    # 构建序列（seq=1）
    X_tr_seq, y_tr_seq = create_sequences(X_train.values, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_ts_seq, y_ts_seq = create_sequences(X_test.values,  y_test,  SEQ_LENGTH, PRED_LENGTH)

    # 验证集
    val_ratio = 0.15
    tr_idx = int(len(X_tr_seq) * (1 - val_ratio))
    X_tr, X_val = X_tr_seq[:tr_idx], X_tr_seq[tr_idx:]
    y_tr, y_val = y_tr_seq[:tr_idx], y_tr_seq[tr_idx:]

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = KANForecaster(
        input_size=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        output_size=PRED_LENGTH,
        grid=GRID,
        k=K
    ).to(device)

    print(model)

    # 训练（完全复用）
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                        patience=PATIENCE, device=device)

    # 滚动预测评估（完全复用）
    rolling_forecast_and_evaluate(
        model, df_test, feature_cols, scaler_y,
        seq_length=SEQ_LENGTH, forecast_horizon=PRED_LENGTH, device=device
    )

    # 可选：KAN 特有的可视化（官方 pykan 自带）
    model.kan.plot()                    # 画网络结构
    model.kan.prune()                   # 自动剪枝
    model.kan = model.kan.prune()       # 更新剪枝后的模型