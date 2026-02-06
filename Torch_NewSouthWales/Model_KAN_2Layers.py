import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils_Non_Tree import *
# 假設 Model_KAN_SCU_base.py 與當前檔案在同一目錄
from Model_KAN_2Layers_base import *


# ────────────────────────────────────────────────
# 設備
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ────────────────────────────────────────────────
# 自製 KAN 模型封裝（兼容 Utils 中的訓練與預測流程）
# ────────────────────────────────────────────────
class KANForecaster2Layers(nn.Module):
    def __init__(self, input_size, hidden_dim=32, output_size=1, device=device):
        super().__init__()
        self.device = device
        self.kan = KANModel(
            input_dim=input_size,      # 特徵數（包含 one-hot + lag + 時間特徵等）
            hidden_dim=hidden_dim,
            output_dim=output_size,    # 預測 1 天
            device=device
        )
        self.softplus = nn.Softplus()  # 保證能耗預測非負

    def forward(self, x):
        # x shape: (batch, seq_length=1, input_size) → 壓成 (batch, input_size)
        x = x.squeeze(1)                      # 去掉 seq 維度
        out = self.kan(x)                     # 自製 KAN 前向
        return self.softplus(out)             # 非負約束


# ────────────────────────────────────────────────
# 主程序（幾乎與 Model_KAN_pykan.py 一致）
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 1           # 自製 KAN 也是純前饋網路，seq=1 即可
    PRED_LENGTH   = 1
    TEST_RATIO    = 0.2
    HIDDEN_DIM    = 32          # 可調整，建議從 16~64 試
    LR            = 0.001       # 自製 KAN 對學習率敏感，可試 3e-4 ~ 2e-3
    EPOCHS        = 300         # 通常收斂比 MLP 快，可設 150~500
    PATIENCE      = 25
    BATCH_SIZE    = 64

    # ── 資料準備（完全復用） ──
    df = load_daily_data()
    (X_train, X_test, y_train, y_test, y_test_orig,
     df_test, scaler_X, scaler_y, feature_cols) = prepare_data(df, LAG_WINDOW, TEST_RATIO)

    print("Features:", feature_cols)
    print(f"Input dimension for KAN-SCU: {len(feature_cols)}")

    # 建立序列（但 seq=1，所以其實退化成普通回歸）
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_ts_seq, y_ts_seq = create_sequences(X_test,  y_test,  SEQ_LENGTH, PRED_LENGTH)

    # 驗證集切分
    val_ratio = 0.15
    tr_idx = int(len(X_tr_seq) * (1 - val_ratio))
    X_tr, X_val = X_tr_seq[:tr_idx], X_tr_seq[tr_idx:]
    y_tr, y_val = y_tr_seq[:tr_idx], y_tr_seq[tr_idx:]

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 模型實例化 ──
    model = KANForecaster2Layers(
        input_size=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        output_size=PRED_LENGTH,
        device=device
    ).to(device)

    print(model)

    # ── 訓練（完全復用） ──
    model = train_model(model, train_loader, val_loader,
                        epochs=EPOCHS, lr=LR,
                        patience=PATIENCE, device=device)

    # ── 真正自回歸滾動預測與評估（完全復用） ──
    rolling_forecast_and_evaluate(
        model, df_test, feature_cols, scaler_y,
        seq_length=SEQ_LENGTH, forecast_horizon=PRED_LENGTH,
        device=device, lag_window=LAG_WINDOW
    )

    # ── 可選：檢查 / 可視化自製 KAN 的 spline（來自 Model_KAN_SCU_base.py） ──
    print("\n=== 第一層 KAN 可視化與活躍檢查 ===")
    print_active_splines(model.kan.l1, threshold=1e-4, name='KAN Layer 1')
    visualize_all_splines(model.kan.l1, layer_name='kan_l1')

    print("\n=== 第二層 KAN 可視化與活躍檢查 ===")
    print_active_splines(model.kan.l2, threshold=1e-4, name='KAN Layer 2')
    visualize_all_splines(model.kan.l2, layer_name='kan_l2')

    # 如果想看訓練後最重要的 spline，可以手動挑選 activity 最大的幾條
    # 例如：visualize_spline(model.kan.l1, input_idx=最重要特徵索引, output_idx=0, ...)