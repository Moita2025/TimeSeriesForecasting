import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")
from ..utils.time_series_utils import get_csv, adf_test, scale_data, grid_search_arima, rolling_forecast
from ..utils.time_series_utils import evaluate_model, plot_compare, plot_resid

# ---------------------------
# 1. 读取数据 & 预处理
# ---------------------------
df = get_csv("../synthetic_load_15min_7200.csv", "15min")

# 选择 'Load_Main' 作为目标变量，并将其他列作为外生变量
exog_columns = ['Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']
exog_ori = df[exog_columns]  # 外生变量
exog_scaled,_ = scale_data(exog_ori)
exog = pd.DataFrame(exog_scaled, columns=exog_ori.columns, index=exog_ori.index)

series = df['Load_Main']  # 目标变量

# 划分训练集和测试集（例如最后 20% 做测试）
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]
exog_train, exog_test = exog[:train_size], exog[train_size:]

print(f"训练集长度: {len(train)}, 测试集长度: {len(test)}")

# ---------------------------
# 2. 平稳性检验（ADF test）
# ---------------------------
print("原始序列 ADF 检验：")
adf_test(train)

# 对外生变量也进行平稳性检验
for col in exog.columns:
    print(f"\n{col} 的 ADF 检验：")
    adf_test(exog_train[col])

# 一次差分（如果需要）
train_diff1 = train.diff().dropna()
exog_diff1 = exog_train.diff().dropna()  # 差分外生变量
print("\n一次差分后 ADF 检验：")
adf_test(train_diff1)

# ---------------------------
# 3. 网格搜索常见 (p,d,q) 组合（推荐范围）
# ---------------------------
p_range = range(0, 5)
d_range = range(0, 2)   # 一般 0~2 就够了
q_range = range(0, 5)

best_aic = float("inf")
best_order = None
best_model = None

best_model, best_order = grid_search_arima(train, p_range, d_range, q_range,
                                           model_type='SARIMAX', exog=exog_train)

# best_order = (0, 1, 1)

# ---------------------------
# 4. 用最佳模型进行滚动预测
# ---------------------------
print(f"使用阶数进行滚动预测: SARIMAX{best_order}\n")

# 先在完整训练集上拟合一次（最耗时的一步只做一次）
print("正在拟合初始模型（请稍候）...")
initial_model = SARIMAX(train, exog=exog_train, order=best_order)
model_fit = initial_model.fit()
print("初始模型拟合完成，开始滚动预测...\n")

def arimax_forecast_func(model, history, exog_train, exog_test, i):
    return model.forecast(steps=1, exog=exog_test.iloc[i:i+1])[0]

# 准备滚动历史（使用 list 比 Series 拼接更快）
history = list(train)
predictions = []

predictions = rolling_forecast(model_fit, train, test, 
                               exog_train=exog_train, exog_test=exog_test,
                               forecast_func=arimax_forecast_func)

# ---------------------------
# 5. 性能评估
# ---------------------------
evaluate_model(test, predictions)

# ---------------------------
# 6. 可视化
# ---------------------------
plot_compare(test, predictions, best_order)

# 残差图
plot_resid(test, predictions)
