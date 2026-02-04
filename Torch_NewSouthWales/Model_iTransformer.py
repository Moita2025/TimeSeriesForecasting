import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iTransformer import iTransformer   # lucidrains 的版本

from Utils_Non_Tree import *

# ────────────────────────────────────────────────
# 設備選擇
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# iTransformer 封裝（lucidrains 版本）
# ────────────────────────────────────────────────
class iTransformerForecaster(nn.Module):
    def __init__(self,
                 num_variates,          # len(feature_cols)
                 lookback_len,
                 pred_len=1,
                 dim=512,
                 depth=6,
                 heads=8,
                 dim_head=64,
                 num_tokens_per_variate=1,
                 dropout=0.1):
        super().__init__()
        
        self.itransformer = iTransformer(
            num_variates = num_variates,
            lookback_len = lookback_len,
            pred_length  = pred_len,               # 可以是 int 或 tuple
            dim          = dim,
            depth        = depth,
            heads        = heads,
            dim_head     = dim_head,
            num_tokens_per_variate = num_tokens_per_variate,
            # 可選：如果你想加 dropout 或其他，可以看 repo 是否支援
        )
        
        # 因為原模型輸出 (batch, pred_len, num_variates)
        # 但我們只想預測 1 個目標，所以加一個線性層投影到 1 維
        self.proj = nn.Linear(num_variates, 1)

    def forward(self, x):
        # x: (batch, seq_len, num_variates)
        out = self.itransformer(x)              # (batch, pred_len, num_variates)
        
        # 只取最後一個預測步長（如果 pred_len > 1 時可調整）
        out = out[:, -1, :]                     # (batch, num_variates)
        
        # 投影到單一目標
        out = self.proj(out)                    # (batch, 1)
        
        return out


# ────────────────────────────────────────────────
# 主程式
# ────────────────────────────────────────────────
if __name__ == "__main__":
    LAG_WINDOW    = 7
    SEQ_LENGTH    = 14          # lookback_len
    PRED_LENGTH   = 1
    TEST_RATIO    = 0.2
    BATCH_SIZE    = 64

    # iTransformer (lucidrains) 建議參數
    DIM           = 512
    DEPTH         = 4           # 先從小開始，資源夠可到 6~8
    HEADS         = 8
    DIM_HEAD      = 64
    NUM_TOKENS    = 1           # 可試 2 或 3 看效果
    DROPOUT       = 0.1
    LR            = 1e-4        # lucidrains 版本通常用較小學習率
    EPOCHS        = 50
    PATIENCE      = 10

    # 資料準備（與之前相同）
    df = load_daily_data()
    (X_train, X_test, y_train, y_test, y_test_orig,
     df_test, scaler_X, scaler_y, feature_cols) = prepare_data(df, LAG_WINDOW, TEST_RATIO)

    print("Features:", feature_cols)
    print(f"Number of input features: {len(feature_cols)}")

    # 建立序列
    X_tr_seq, y_tr_seq = create_sequences(X_train.values, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_ts_seq, y_ts_seq = create_sequences(X_test.values,  y_test,  SEQ_LENGTH, PRED_LENGTH)

    # 分驗證集
    val_ratio = 0.15
    tr_idx = int(len(X_tr_seq) * (1 - val_ratio))
    X_tr, X_val = X_tr_seq[:tr_idx], X_tr_seq[tr_idx:]
    y_tr, y_val = y_tr_seq[:tr_idx], y_tr_seq[tr_idx:]

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = iTransformerForecaster(
        num_variates = len(feature_cols),
        lookback_len = SEQ_LENGTH,
        pred_len     = PRED_LENGTH,
        dim          = DIM,
        depth        = DEPTH,
        heads        = HEADS,
        dim_head     = DIM_HEAD,
        num_tokens_per_variate = NUM_TOKENS,
        dropout      = DROPOUT
    ).to(device)

    print(model)

    # 訓練（使用你原本的 train_model）
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=device
    )

    # 滾動預測（注意 forward 輸出已是 (batch,1)）
    rolling_forecast_and_evaluate(
        model,
        df_test,
        feature_cols,
        scaler_y,
        seq_length=SEQ_LENGTH,
        forecast_horizon=PRED_LENGTH,
        device=device
    )