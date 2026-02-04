import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_load_data(
    n_samples=6000,          # 总长度
    freq='15min',            # 时间分辨率：'15min', '30min', 'H', 'D'
    random_seed=42
):
    np.random.seed(random_seed)
    
    # 时间索引
    t = np.arange(n_samples)
    
    # 基本趋势（缓慢上升/下降）
    trend = 0.0003 * t**1.4 + 300   # 轻微非线性增长
    
    # 年周期（假设一年365.25天）
    yearly = 150 * np.sin(2 * np.pi * t / (365.25 * 24 * 4))   # 15分钟分辨率下的一年周期
    
    # 周周期（7天）
    weekly = 80 * np.sin(2 * np.pi * t / (7 * 24 * 4))
    
    # 日周期（一天24小时）
    daily = 220 * np.sin(2 * np.pi * t / (24 * 4))              # 主日周期
    daily += 90 * np.sin(4 * np.pi * t / (24 * 4))              # 二次谐波
    daily += 40 * np.sin(6 * np.pi * t / (24 * 4) + 0.7)        # 三次谐波 + 相位偏移
    
    # 早晚高峰微调（更真实）
    morning_peak = 60 * np.exp(-((t % (24*4) - 28)**2)/ (2*8**2))
    evening_peak = 80 * np.exp(-((t % (24*4) - 75)**2)/ (2*10**2))
    
    # 随机噪声（不同层级）
    noise = np.random.normal(0, 18, n_samples)                  # 基础白噪声
    noise += np.random.normal(0, 12, n_samples) * (1 + np.sin(2*np.pi*t/(24*4)))  # 日内波动变化
    
    # 组合
    signal = trend + yearly + weekly + daily + morning_peak + evening_peak + noise
    
    # 加入少量异常/尖峰（可选，模拟突发事件）
    spike_idx = np.random.choice(n_samples, size=8, replace=False)
    signal[spike_idx] += np.random.uniform(120, 280, 8)
    
    # 生成多变量（这里简单用 3~5 个相关变量）
    data = np.column_stack([
        signal,
        signal * 0.65 + np.random.normal(0, 25, n_samples),      # 区域A
        signal * 1.15 + np.random.normal(0, 32, n_samples),      # 区域B（工业区）
        signal * 0.92 + np.random.normal(0, 22, n_samples),      # 区域C
        0.4 * signal + 120 + np.random.normal(0, 18, n_samples)  # 温度（弱相关）
    ])
    
    # 做成DataFrame，方便后续使用
    df = pd.DataFrame(
        data,
        index=pd.date_range(start='2023-01-01', periods=n_samples, freq=freq),
        columns=['Load_Main', 'Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']
    )
    
    return df


# 使用示例
if __name__ == '__main__':
    df = generate_synthetic_load_data(n_samples=7200, freq='15min')  # ≈5个月 15分钟数据
    
    print(df.shape)
    print(df.head())
    
    # 可视化前两周
    df.iloc[:24*4*14].plot(figsize=(14,6), subplots=True, sharex=True)
    plt.tight_layout()
    plt.show()
    
    # 保存（可选）
    df.to_csv("synthetic_load_15min_7200.csv", float_format="%.3f")