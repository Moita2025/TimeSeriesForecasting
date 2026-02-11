import torch
from typing import List, Dict, Optional

from data_module.nsw_datamodule_halfhour import NSWHalfHourDataModule
from trainer import train_model
from rolling_eval import predict_rolling_windows, compute_forecast_metrics
from tasks.next_48h_forecast import Next48HalfHoursForecastTask
from config import model_configs
from models import create_model


def prepare_data_and_task(
    data_file: str,
    seq_len: int = 336,
    horizon: int = 48,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
) -> tuple[NSWHalfHourDataModule, Next48HalfHoursForecastTask]:
    """準備資料與任務物件（只執行一次）"""
    datamodule = NSWHalfHourDataModule(
        data_path=data_file,
        seq_len=seq_len,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )
    datamodule.prepare_data()

    task = Next48HalfHoursForecastTask({
        "seq_len": seq_len,
        "horizon": horizon,
        # 其他 task 需要的設定可以從這裡傳入
    })

    return datamodule, task


def get_model_config(
    model_type: str,
    common_config: Dict,
) -> Optional[Dict]:
    """取得單一模型的完整 config，若不存在則回傳 None"""
    if model_type not in model_configs:
        print(f"⚠️ 跳過未知模型：{model_type}")
        return None

    return {**common_config, **model_configs[model_type]}


def run_single_model(
    model_type: str,
    datamodule: NSWHalfHourDataModule,
    task: Next48HalfHoursForecastTask,
    common_config: Dict,
    device: torch.device,
) -> None:
    """對單一模型執行：建立 → 訓練 → 滾動評估 → 後處理"""
    config = get_model_config(model_type, common_config)
    if config is None:
        return

    print(f"\n{'='*60}")
    print(f"開始訓練模型：{model_type.upper()}")
    print(f"序列長度: {config['seq_len']}   預測長度: {config['horizon']}")
    print(f"{'='*60}\n")

    # 建立模型
    model = create_model(
        model_type=model_type,
        input_size=datamodule.input_size,
        **config
    ).to(device)

    print(f"模型：{model_type.upper()}")
    print(f"參數量約: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"裝置：{device}\n")

    # 訓練
    trained_model = train_model(
        model=model,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"],
        grad_clip=config.get("grad_clip", 1.0),
        device=device,
    )

    # ── 预测 + 反归一化 ────────────────────────────────────────
    raw_preds_phys, acts_phys = predict_rolling_windows(
        model=trained_model,
        X_test_seq=datamodule.X_test_seq,
        y_test_seq=datamodule.y_test_seq,
        scaler_demand=datamodule.scaler_demand,
        scaler_price=datamodule.scaler_price,
        device=device,
        batch_size=128,
    )

    # ── 任务特定的后处理（bias correction, clip 等） ────────────
    preds_processed = task.postprocess_predictions(raw_preds_phys)

    metrics_before = compute_forecast_metrics(
        predictions=raw_preds_phys,
        actuals=acts_phys,
        mask_demand_min=task.eval_mask["demand"]["min_value"],
        mask_price_min=task.eval_mask["price"]["min_value"],
        print_summary=True,               # 或设为 False，交给 task.format_metrics 统一打印
    )

    # ── 计算最终指标（基于后处理后的预测） ─────────────────────
    metrics_after = compute_forecast_metrics(
        predictions=preds_processed,
        actuals=acts_phys,
        mask_demand_min=task.eval_mask["demand"]["min_value"],
        mask_price_min=task.eval_mask["price"]["min_value"],
        print_summary=True,               # 或设为 False，交给 task.format_metrics 统一打印
    )

    # 顯示結果（或存檔、畫圖）
    # print("\n" + task.format_metrics(metrics))

    # 可選：儲存預測結果、畫圖等
    # save_predictions(preds_processed, acts_phys, f"results/{model_type}/")
    # plot_sample_forecasts(preds_processed, acts_phys, num_samples=8, save_dir=f"figures/{model_type}/")


def main():
    # ── 共用設定 ───────────────────────────────────────────────
    common_config = {
        "data_file":      "data_processed/nsw_halfhour_201207_201306.csv",
        "seq_len":        336,
        "horizon":        48,
        "train_ratio":    0.7,
        "val_ratio":      0.15,
        "batch_size":     64,
        "epochs":         100,
        "lr":             0.0008,
        "patience":       12,
        "grad_clip":      1.0,
    }

    # ── 要跑的模型列表（容易增減） ──────────────────────────────
    model_types: List[str] = [
        "cnn_gru",
        "bilstm",
        "am_bilstm",
        "tpa_bilstm",
        "itransformer",
        "dnn"
    ]

    # ── 準備資料（只需一次） ────────────────────────────────────
    datamodule, task = prepare_data_and_task(
        data_file=common_config["data_file"],
        seq_len=common_config["seq_len"],
        horizon=common_config["horizon"],
        train_ratio=common_config["train_ratio"],
        val_ratio=common_config["val_ratio"],
        batch_size=common_config["batch_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # ── 逐一訓練與評估每個模型 ─────────────────────────────────
    for model_type in model_types:
        try:
            run_single_model(
                model_type=model_type,
                datamodule=datamodule,
                task=task,
                common_config=common_config,
                device=device,
            )
        except Exception as e:
            print(f"\n❌ 模型 {model_type} 執行失敗：{e}\n")
            import traceback
            traceback.print_exc()
            print("-"*80)


if __name__ == "__main__":
    main()