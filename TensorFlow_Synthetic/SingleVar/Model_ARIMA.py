import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
from ..utils.time_series_utils import get_csv, adf_test, grid_search_arima, rolling_forecast
from ..utils.time_series_utils import evaluate_model, plot_compare, plot_resid, plot_acf_pacf

# ---------------------------
# 1. 读取数据 & 预处理
# ---------------------------
df = get_csv("../synthetic_load_15min_7200.csv", "15min")

# 我们先只用主负荷列做实验（可后续扩展到多变量）
series = df['Load_Main'].copy()

# 划分训练/测试集（例如最后 20% 做测试）
train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print(f"训练集长度: {len(train)}, 测试集长度: {len(test)}")

print("原始序列 ADF 检验：")
adf_test(train)

# 一次差分
train_diff1 = train.diff().dropna()
print("\n一次差分后 ADF 检验：")
adf_test(train_diff1)

# ---------------------------
# 3. 绘制 ACF / PACF 图（帮助判断 p,q）
# ---------------------------
plot_acf_pacf(train_diff1)

# ---------------------------
# 4. 网格搜索常见 (p,d,q) 组合（推荐范围）
# ---------------------------
p_range = range(0, 5)
d_range = range(0, 2)   # 一般 0~2 就够了
q_range = range(0, 5)

best_model, best_order = grid_search_arima(train, p_range, d_range, q_range)

# ---------------------------
# 5. 用最佳模型进行预测
# ---------------------------
# 滚动预测（一步步预测，更接近真实场景）
print(f"使用阶数进行滚动预测: ARIMA{best_order}\n")

# ── 推薦的高效率滚动预测写法 ───────────────────────────────────────

# 先在完整训练集上拟合一次（最耗时的一步只做一次）
print("正在拟合初始模型（请稍候）...")
initial_model = ARIMA(train, order=best_order)
model_fit = initial_model.fit()
print("初始模型拟合完成，开始滚动预测...\n")

def arima_forecast_func(model, history, exog_train, exog_test, i):
    forecast_result = model.forecast(steps=1)
    yhat = forecast_result.iloc[0] if isinstance(forecast_result, pd.Series) else forecast_result[0]
    return yhat

# 准备滚动历史（使用 list 比 Series 拼接更快）
history = list(train)
predictions = []

predictions = rolling_forecast(model_fit, train, test, forecast_func=arima_forecast_func)

# ── 性能评估 ────────────────────────────────────────────────────────
evaluate_model(test, predictions)

# ── 可视化 ───────────────────────────────────────────────────────────
plot_compare(test, predictions, best_order)

# 残差图
plot_resid(test, predictions)