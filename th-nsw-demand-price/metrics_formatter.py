from typing import Dict, List
import pandas as pd


def format_metrics_to_html(
    all_results: List[Dict],
    output_file: str = "./results/nsw_1year_with_before_after.html",
    title: str = "多模型預測性能比較（前處理 vs 後處理，預測長度：48步）"
) -> None:
    """
    將多個模型的前/後處理 metrics 轉成 HTML 表格
    
    all_results 格式示例：
    [
        {
            "model": "CNN_GRU",
            "stage": "Before postprocess",
            "demand": {"mae": 1053.68, "rmse": 1298.99, "mape": 11.81, ...},
            "price":  {"mae": 7.92,  "rmse": 13.02,  "mape": 12.39, ...}
        },
        {
            "model": "CNN_GRU",
            "stage": "After postprocess",
            "demand": {...},
            "price": {...}
        },
        ...
    ]
    """
    rows = []
    
    for res in all_results:
        model = res["model"]
        stage = res["stage"]
        
        d = res["demand"]
        p = res["price"]
        
        rows.append({
            "模型名稱": model,
            "處理階段": stage,
            "Demand MAE (MW)": f"{d['mae']:.2f}",
            "Demand RMSE (MW)": f"{d['rmse']:.2f}",
            "Demand MAPE": f"{d['mape']:.2f}%" if not pd.isna(d['mape']) else "N/A",
            "Demand Mean actual": f"{d['mean_actual']:.2f}",
            "Demand Mean pred": f"{d['mean_pred']:.2f}",
            "Price MAE ($/MWh)": f"{p['mae']:.2f}",
            "Price RMSE ($/MWh)": f"{p['rmse']:.2f}",
            "Price MAPE": f"{p['mape']:.2f}%" if not pd.isna(p['mape']) else "N/A",
            "Price Mean actual": f"{p['mean_actual']:.2f}",
            "Price Mean pred": f"{p['mean_pred']:.2f}",
        })

    df = pd.DataFrame(rows)
    
    # 排序：相同模型的 before 在上，after 在下
    df["model_order"] = df["模型名稱"].map({m: i for i, m in enumerate(sorted(set(df["模型名稱"])))})
    df = df.sort_values(["model_order", "處理階段"], ascending=[True, False])  # After 先？或改成 True 讓 Before 先
    df = df.drop(columns=["model_order"])

    # 轉 HTML，保留樣式類似你原本的 nsw_1year.html
    html = df.to_html(
        index=False,
        border=1,
        classes="table table-striped",
        float_format="%.2f",
        na_rep="N/A"
    )

    # 添加自訂 CSS 與 caption
    full_html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
            max-width: 1200px;
            margin: 20px auto;
            font-family: Arial, sans-serif;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .model-name {{
            background-color: #e6f3ff;
            font-weight: bold;
            vertical-align: middle;
        }}
        caption {{
            font-size: 1.3em;
            padding: 12px;
            font-weight: bold;
            caption-side: top;
        }}
        .stage-before {{ background-color: #fff3e6; }}
        .stage-after  {{ background-color: #e6ffe6; }}
    </style>
</head>
<body>
<h2 style="text-align:center;">時間序列預測模型評估結果比較 - TH-NSW Demand & Price</h2>
{html.replace('<table', f'<table><caption>{title}</caption').replace('<tr>', '<tr>').replace(
    '<td>', '<td class="stage-before">' if 'Before' in df["處理階段"].values else '<td>'
)}
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print(f"已儲存比較表格 → {output_file}")