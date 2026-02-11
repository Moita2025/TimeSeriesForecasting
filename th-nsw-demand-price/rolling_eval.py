import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import torch
from typing import Dict, Tuple


def predict_rolling_windows(
    model: torch.nn.Module,
    X_test_seq: np.ndarray,           # (n_windows, seq_len, n_features) scaled
    y_test_seq: np.ndarray,           # (n_windows, horizon, 2) scaled
    scaler_demand,
    scaler_price,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 128,
    desc: str = "Batch inference on test set",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对测试集的所有滚动窗口进行批量预测 + 反归一化
    只返回物理单位的预测和真实值，不计算指标

    返回:
        predictions: (n_windows, horizon, 2) physical unit
        actuals:     (n_windows, horizon, 2) physical unit
    """
    model.eval()
    model.to(device)

    all_pred = []
    all_true = []

    n_samples = len(X_test_seq)
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=desc):
            end = min(i + batch_size, n_samples)
            Xb = torch.from_numpy(X_test_seq[i:end]).float().to(device)

            pred_scaled = model(Xb).cpu().numpy()          # (b, horizon, 2)

            # 反归一化 demand
            pred_d = scaler_demand.inverse_transform(
                pred_scaled[..., 0].reshape(-1, 1)
            ).reshape(-1, pred_scaled.shape[1])

            # 反归一化 price
            pred_p = scaler_price.inverse_transform(
                pred_scaled[..., 1].reshape(-1, 1)
            ).reshape(-1, pred_scaled.shape[1])

            # 真实值反归一化
            true_d = scaler_demand.inverse_transform(
                y_test_seq[i:end, :, 0].reshape(-1, 1)
            ).reshape(-1, y_test_seq.shape[1])

            true_p = scaler_price.inverse_transform(
                y_test_seq[i:end, :, 1].reshape(-1, 1)
            ).reshape(-1, y_test_seq.shape[1])

            all_pred.append(np.stack([pred_d, pred_p], axis=-1))
            all_true.append(np.stack([true_d, true_p], axis=-1))

    predictions = np.concatenate(all_pred, axis=0)   # (total_windows, horizon, 2)
    actuals     = np.concatenate(all_true, axis=0)

    return predictions, actuals

def compute_forecast_metrics(
    predictions: np.ndarray,          # (n_windows, horizon, 2) physical
    actuals: np.ndarray,              # (n_windows, horizon, 2) physical
    mask_demand_min: float = 1000.0,
    mask_price_min: float = 5.0,
    print_summary: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    计算需求和价格的 MAE / RMSE / MAPE / mean_actual / mean_pred
    支持 masked MAPE

    返回:
        metrics: {
            "demand": {"mae": ..., "rmse": ..., "mape": ..., "mean_actual": ..., "mean_pred": ...},
            "price":  {...}
        }
    """
    # 展平所有步驟
    demand_true = actuals[..., 0].ravel()
    demand_pred = predictions[..., 0].ravel()
    price_true  = actuals[..., 1].ravel()
    price_pred  = predictions[..., 1].ravel()

    # Demand
    mae_d  = mean_absolute_error(demand_true, demand_pred)
    rmse_d = np.sqrt(mean_squared_error(demand_true, demand_pred))

    mask_d = np.abs(demand_true) > mask_demand_min
    mape_d = np.mean(np.abs((demand_true - demand_pred)[mask_d] / demand_true[mask_d])) * 100 \
        if mask_d.sum() > 0 else np.nan

    # Price
    mae_p  = mean_absolute_error(price_true, price_pred)
    rmse_p = np.sqrt(mean_squared_error(price_true, price_pred))

    mask_p = np.abs(price_true) > mask_price_min
    mape_p = np.mean(np.abs((price_true - price_pred)[mask_p] / price_true[mask_p])) * 100 \
        if mask_p.sum() > 0 else np.nan

    metrics = {
        "demand": {
            "mae":  mae_d,
            "rmse": rmse_d,
            "mape": mape_d,
            "mean_actual": float(demand_true.mean()),
            "mean_pred":   float(demand_pred.mean()),
        },
        "price": {
            "mae":  mae_p,
            "rmse": rmse_p,
            "mape": mape_p,
            "mean_actual": float(price_true.mean()),
            "mean_pred":   float(price_pred.mean()),
        }
    }

    if print_summary:
        print("\n" + "="*70)
        print("Evaluation (all 48 steps, physical units):")
        print("-"*70)
        print("Demand:")
        print(f"  MAE  = {mae_d:8.2f} MW")
        print(f"  RMSE = {rmse_d:8.2f} MW")
        print(f"  MAPE = {mape_d:6.2f}% (masked)" if not np.isnan(mape_d) else "  MAPE = N/A")
        print(f"  Mean actual: {demand_true.mean():8.2f}   Mean pred: {demand_pred.mean():8.2f}")
        print()
        print("Price (RRP):")
        print(f"  MAE  = {mae_p:8.2f} $/MWh")
        print(f"  RMSE = {rmse_p:8.2f} $/MWh")
        print(f"  MAPE = {mape_p:6.2f}% (masked)" if not np.isnan(mape_p) else "  MAPE = N/A")
        print(f"  Mean actual: {price_true.mean():8.2f}   Mean pred: {price_pred.mean():8.2f}")
        print("="*70 + "\n")

    return metrics