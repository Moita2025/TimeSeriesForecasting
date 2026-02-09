import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tasks.base_task import BaseForecastTask
from utils import TimeSeriesDataset   # 建议后续启用

if __name__ == "__main__":
    # ── 任务与模型选择 ────────────────────────────────────────
    task_key   = "next_day_48halfhours"
    model_key  = "Model_CNN_GRU"

    task_class_dict = {
        "next_day_48halfhours": "NextDay48HalfHoursRRP"
    }
    model_class_dict = {
        "Model_CNN_GRU": "CNNGRUForecaster"
    }

    print(f"任务：{task_key}")
    print(f"模型：{model_key}")

    # 动态导入
    TaskClass = getattr(__import__(f"tasks.{task_key}", fromlist=[task_key]), 
                        task_class_dict[task_key])
    
    ModelClass = getattr(__import__(f"models.{model_key}", fromlist=[model_key]), 
                         model_class_dict[model_key])

    # ── 配置 ──────────────────────────────────────────────────
    config = {
        "data_file"      : "data_processed/nsw_halfhour_201207_201306.csv",
        "target_col"     : "RRP",           # 仅作兼容，实际以 task.target_cols 为准
        "seq_len"        : 336,             # 7天
        "horizon"        : 48,
        "train_ratio"    : 0.8,
        "batch_size"     : 64,
        "epochs"         : 100,
        "lr"             : 0.0008,
        "patience"       : 12,
        "hidden_size"    : 128,
        "num_gru_layers" : 2,
        "num_cnn_filters": 64,
        "kernel_size"    : 3,
        "dropout"        : 0.15,
        "bidirectional"  : False,
        "use_pool"       : False,
    }

    # ── 实例化任务 & 准备数据 ────────────────────────────────
    task = TaskClass(config)
    X_tr, y_tr, X_ts, y_ts, extra, X_ts_raw = task.prepare_data()

    # 重要：更新 input_size（之前是 None）
    config["input_size"] = X_tr.shape[-1]
    print(f"输入特征维度: {config['input_size']}")

    # ── 数据集 & DataLoader ───────────────────────────────────
    # 方式一：使用 utils 里的 TimeSeriesDataset（推荐）
    # train_dataset = TimeSeriesDataset(X_tr, y_tr)
    # test_dataset  = TimeSeriesDataset(X_ts, y_ts)

    # 方式二：临时自己写（当前最快能跑通的方式）
    class SimpleTSDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    full_dataset = SimpleTSDataset(X_tr, y_tr)
    
    train_size = int(0.85 * len(full_dataset))   # 在训练集中再分验证
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)

    # ── 模型 & 训练 ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(config).to(device)

    print(f"使用设备：{device}")

    # 使用 base_task 里已有的训练函数（非常方便）
    trained_model = BaseForecastTask.train_model(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        epochs         = config["epochs"],
        lr             = config["lr"],
        patience       = config["patience"],
        device         = device
    )

    # ── 滚动预测与评估 ───────────────────────────────────────
    metrics, preds, acts = task.rolling_forecast_and_evaluate(
        model   = trained_model,
        X_test  = X_ts_raw,
        y_test  = y_ts,
        extra   = extra,
        device  = device
    )

    print("滚动预测评估完成。")