from typing import Dict, Any, Optional
import numpy as np

class Next48HalfHoursForecastTask:
    """
    下一天 48 個半小時點的 NSW 電力需求與價格預測任務
    
    這個類只負責定義任務的「本質」與「偏好」，不處理數據與訓練細節。
    """

    def __init__(self, config: Dict[str, Any]):
        self.target_cols = ["TOTALDEMAND", "RRP"]
        self.target_names = {"TOTALDEMAND": "Demand (MW)", "RRP": "RRP ($/MWh)"}
        
        self.horizon = config.get("horizon", 48)
        self.seq_len = config.get("seq_len", 336)
        
        # 評估相關偏好
        self.eval_mask = {
            "demand": {"min_value": 1000.0},   # MAPE 時忽略低於此值的樣本
            "price":  {"min_value": 5.0}
        }
        
        self.eval_metrics = ["mae", "rmse", "mape"]
        
        # 可選：後處理規則
        self.postprocess_rules = {
            "demand": {"clip_min": 0.0},      # 需求不應該為負
            "price":  {}                      # 價格允許負值（真實情況有負價）
        }

    def get_task_signature(self) -> str:
        """用來 log 或顯示的任務描述"""
        return f"Next {self.horizon} half-hours forecast: Demand & RRP"

    def postprocess_predictions(
        self,
        predictions: np.ndarray,     # shape: (n_samples, horizon, 2)
        actuals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        對物理單位的預測結果做任務特定的後處理
        
        目前很簡單，未來可加平滑、clip、異常修正等
        """
        preds = predictions.copy()
        
        # demand clip
        if "demand" in self.postprocess_rules:
            min_val = self.postprocess_rules["demand"].get("clip_min", 0.0)
            preds[..., 0] = np.maximum(preds[..., 0], min_val)
            
        # price 可以保留負值，不做 clip
        
        return preds

    def format_metrics(self, metrics: Dict) -> str:
        """把 metrics 格式化成易讀的字串（可供 print 或 log）"""
        lines = []
        lines.append("Evaluation Results:")
        lines.append("-" * 60)
        
        for target, target_metrics in metrics.items():
            name = self.target_names.get(target, target)
            lines.append(f"{name}:")
            for m in self.eval_metrics:
                val = target_metrics.get(m, np.nan)
                if m == "mape":
                    lines.append(f"  {m.upper():4} = {val:6.2f}%")
                else:
                    unit = "MW" if target == "TOTALDEMAND" else "$/MWh"
                    lines.append(f"  {m.upper():4} = {val:8.2f} {unit}")
            lines.append(f"  Mean actual: {target_metrics['mean_actual']:8.2f}")
            lines.append(f"  Mean pred  : {target_metrics['mean_pred']:8.2f}")
            lines.append("")
            
        return "\n".join(lines)