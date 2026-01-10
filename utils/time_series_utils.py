import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体（若系统没有，则 matplotlib 会自动 fallback）
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------
# 1. ADF 检验函数
# ------------------------------
def adf_test(ts):
    result = adfuller(ts)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("→ 序列在5%显著性水平下拒绝原假设 → 平稳")
    else:
        print("→ 序列可能不平稳，需要差分")


# ------------------------------
# 2. 数据标准化
# ------------------------------

def scale_data(df, scaler_type='minmax', exclude_columns=None, is_dict=True):

    # 选择数值列
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # 如果指定排除列，从数值列中移除它们（忽略不存在的列）
    if exclude_columns is not None:
        exclude_columns = set(exclude_columns) & set(numeric_cols)
        target_cols = [col for col in numeric_cols if col not in exclude_columns]
    else:
        target_cols = numeric_cols

    if not target_cols:
        return df.copy(), None
    
    # 为每个特征创建独立 Scaler
    scaled_values = df[target_cols].copy()
    processed_df = df.copy()

    if is_dict:
        # ======== 每列独立 scaler（推荐） ========
        scalers = {}
        for col in target_cols:
            if scaler_type.lower() == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type.lower() == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError("scaler_type must be 'minmax' or 'standard'")

            # 对单列进行 fit & transform
            scaled_values[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    else:
        # ======== 所有目标列共用一个 scaler ========
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        
        # 整体 fit & transform
        scaled_values = pd.DataFrame(
            scaler.fit_transform(scaled_values),
            index=scaled_values.index,
            columns=scaled_values.columns
        )
        scalers = scaler  # 单个 scaler

    # 创建处理后的 DataFrame
    processed_df = df.copy()
    processed_df[target_cols] = scaled_values

    return processed_df, scalers  # 返回字典形式的 Scaler

# ------------------------------
# 3. 网格搜索优化 p, d, q
# ------------------------------
def grid_search_arima(ts, p_range, d_range, q_range, model_type='ARIMA', exog=None):
    best_aic = float("inf")
    best_order = None
    best_model = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    if model_type == 'ARIMA':
                        model = ARIMA(ts, order=(p, d, q))
                    elif model_type == 'SARIMAX':
                        model = SARIMAX(ts, exog=exog, order=(p, d, q))
                    else:
                        raise ValueError("Unsupported model_type")
                    model_fit = model.fit()
                    current_aic = model_fit.aic
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = (p, d, q)
                        best_model = model_fit
                    print(f"{model_type}{p,d,q} AIC = {current_aic:.2f}")
                except Exception as e:
                    print(f"Error fitting {model_type}{p,d,q}: {e}")
                    continue
    print(f"\n最佳模型: {model_type}{best_order} AIC = {best_aic:.2f}")
    return best_model, best_order


# ------------------------------
# 4. 滚动预测函数（支持匿名函数作为参数）
# ------------------------------
def rolling_forecast(model, train, test, 
                     exog_train=None, exog_test=None, 
                     steps=1, forecast_func=None, 
                     refit=False, refit_step=96):
    """
    实现滚动预测，可以传入匿名函数来定制不同的预测方式
    """
    history = list(train)
    predictions = []
    
    for i in tqdm(range(len(test)), desc="滚动预测进度"):
        if forecast_func:
            forecast_result = forecast_func(model, history, exog_train, exog_test, i)
        else:
            forecast_result = model.forecast(steps=steps, exog=exog_test.iloc[i:i+1])
        
        predictions.append(forecast_result)
        new_obs = test.iloc[i]

        # 将最新的真实值加入模型（refit=False 提高效率）
        if refit and ((i + 1) % refit_step == 0 and (i + 1) < len(test)):
            exog_new = exog_test.iloc[i:i+1] if exog_test is not None else None
            model = model.append([new_obs], exog=exog_new)
        else:
            # 更新模型时需要同时提供新的外生变量
            if exog_test is not None:
                model = model.append([new_obs], exog=exog_test.iloc[i:i+1], refit=False)
            else:
                model = model.append([new_obs], refit=False)
        
        # 更新历史记录（用于后续预测）
        history.append(test.iloc[i])
    
    return pd.Series(predictions, index=test.index)

def get_csv(file_name, scale = "15min"):
    
    file_path = file_name  # 改成你的实际路径
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.freq = scale  # 重要！告诉 pandas 频率

    return df

def evaluate_model(test, predictions):

    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    print("\n预测性能：")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

def plot_acf_pacf(train_diff1):

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(train_diff1, lags=60, ax=axes[0])
    axes[0].set_title('ACF - 一次差分')

    plot_pacf(train_diff1, lags=60, method='ywm', ax=axes[1])
    axes[1].set_title('PACF - 一次差分')

    plt.tight_layout()
    plt.show()

def plot_compare(test, predictions, best_order):

    plt.figure(figsize=(15, 6))
    plt.plot(test.index, test, label='真实值', alpha=0.7, linewidth=1)
    plt.plot(test.index, predictions, label='ARIMA 预测', color='orange', linestyle='--', linewidth=1.5)
    plt.title(f'测试集预测对比 - ARIMA{best_order} (增量更新)', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_resid(test, predictions):
    
    residuals = test - predictions
    plt.figure(figsize=(15, 4))
    plt.plot(residuals, color='purple', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', alpha=0.4)
    plt.title('残差图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()