import torch
from data_module.nsw_datamodule_halfhour import NSWHalfHourDataModule
from models.cnn_gru import CNNGRUForecaster
from trainer import train_model
from rolling_eval import evaluate_rolling_forecast
from tasks.next_48h_forecast import Next48HalfHoursForecastTask

def main():
    # ── 配置（之後建議改 yaml 或 dataclass） ─────────────────────────────
    config = {
        "data_file":      "data_processed/nsw_halfhour_201207_201306.csv",
        "seq_len":        336,
        "horizon":        48,
        "train_ratio":    0.7,
        "val_ratio":      0.15,
        "batch_size":     64,
        "epochs":         100,
        "lr":             0.0008,
        "patience":       12,

        # 模型參數
        "hidden_size":    128,
        "num_gru_layers": 2,
        "num_cnn_filters":64,
        "kernel_size":    3,
        "dropout":        0.15,
        "bidirectional":  False,
        "use_pool":       False,
    }

    print("=== NSW 下一天 48 半小時 需求與價格預測 ===")
    print(f"序列長度: {config['seq_len']}   預測長度: {config['horizon']}")

    # ── 1. 資料模組 ──────────────────────────────────────────────────
    datamodule = NSWHalfHourDataModule(
        data_path     = config["data_file"],
        seq_len       = config["seq_len"],
        horizon       = config["horizon"],
        train_ratio   = config["train_ratio"],
        val_ratio     = config["val_ratio"],
        batch_size    = config["batch_size"],
    )
    datamodule.prepare_data()

    # ── 2. 模型 ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNGRUForecaster(
        input_size      = datamodule.input_size,
        hidden_size     = config["hidden_size"],
        num_gru_layers  = config["num_gru_layers"],
        num_cnn_filters = config["num_cnn_filters"],
        kernel_size     = config["kernel_size"],
        dropout         = config["dropout"],
        bidirectional   = config["bidirectional"],
        use_pool        = config["use_pool"],
        horizon         = config["horizon"],
        n_targets       = 2,   # demand + price
    ).to(device)

    print(f"模型參數量約: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"使用裝置: {device}")

    # ── 3. 訓練 ──────────────────────────────────────────────────────
    trained_model = train_model(
        model        = model,
        train_loader = datamodule.train_loader,
        val_loader   = datamodule.val_loader,
        epochs       = config["epochs"],
        lr           = config["lr"],
        patience     = config["patience"],
        grad_clip    = 1.0,
        device       = device,
    )

    # ── 4. 評估 ──────────────────────────────────────────────────────
    task = Next48HalfHoursForecastTask(config)

    metrics, preds_phys, acts_phys = evaluate_rolling_forecast(
        model         = trained_model,
        X_test_seq    = datamodule.X_test_seq,
        y_test_seq    = datamodule.y_test_seq,
        scaler_demand = datamodule.scaler_demand,
        scaler_price  = datamodule.scaler_price,
        device        = device,
        batch_size    = 128,
        mask_demand_min = task.eval_mask["demand"]["min_value"],
        mask_price_min  = task.eval_mask["price"]["min_value"],
    )

    # 任務特定後處理（如果有 clip 等規則）
    preds_processed = task.postprocess_predictions(preds_phys)

    # 格式化輸出
    print("\n" + task.format_metrics(metrics))

    # 未來可加：
    # save_predictions(preds_processed, acts_phys, "results/")
    # plot_sample_forecasts(preds_processed, acts_phys, num_samples=8)


if __name__ == "__main__":
    main()