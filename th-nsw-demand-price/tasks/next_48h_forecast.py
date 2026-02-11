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
        # ── 新增：经验性 bias correction（模型普遍低估 ≈10%） ──
        self.postprocess_rules = {
            "demand": {
                "clip_min": 0.0,
                "multiplicative_bias": 1.10,   # ← 这里改成你实际观察到的比例
            },
            "price": {
                "multiplicative_bias": 1.10,   # 如果价格也低估就一起开，否则删掉这行
            }
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
        
        # Demand
        if "demand" in self.postprocess_rules:
            rules = self.postprocess_rules["demand"]
            if "multiplicative_bias" in rules:
                preds[..., 0] *= rules["multiplicative_bias"]          # ← 关键一行
            if "clip_min" in rules:
                preds[..., 0] = np.maximum(preds[..., 0], rules["clip_min"])

        # Price
        if "price" in self.postprocess_rules:
            rules = self.postprocess_rules["price"]
            if "multiplicative_bias" in rules:
                preds[..., 1] *= rules["multiplicative_bias"]          # ← 关键一行
        
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