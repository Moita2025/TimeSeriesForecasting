# Time Series Prediction Models

A collection of time series forecasting models for both univariate and multivariate scenarios, implemented in Python. The project includes various deep learning and statistical models tested on synthetic electrical load data.

## Features

- **Univariate Time Series Models**:
  - ARIMA
  - BiLSTM
  - Attention Mechanism BiLSTM (AM-BiLSTM)
  - Temporal Pattern Attention BiLSTM (TPA-BiLSTM)
  - CNN-GRU hybrid model

- **Multivariate Time Series Models**:
  - ARIMAX
  - BiLSTM
  - Attention Mechanism BiLSTM (AM-BiLSTM)
  - Temporal Pattern Attention BiLSTM (TPA-BiLSTM)
  - CNN-GRU hybrid model

- **Utility Functions**:
  - ADF stationarity test
  - Data normalization/standardization
  - Rolling window prediction
  - Time series visualization (ACF, PACF, residuals)

## Requirements

The project uses Python 3.x and requires the following packages:
- numpy
- pandas
- scikit-learn
- statsmodels
- tensorflow/keras
- matplotlib
- seaborn

---

# 时间序列预测模型

本项目包含多种时间序列预测模型，适用于单变量和多变量预测场景，所有模型均使用Python实现。项目包含各种深度学习和统计模型，并使用合成电力负荷数据进行测试。

## 功能特点

- **单变量时间序列模型**：
    - ARIMA
    - BiLSTM
    - 注意力机制BiLSTM (AM-BiLSTM)
    - 时间模式注意力BiLSTM (TPA-BiLSTM)
    - CNN-GRU混合模型

- **多变量时间序列模型**：
    - ARIMAX
    - BiLSTM
    - 注意力机制BiLSTM (AM-BiLSTM)
    - 时间模式注意力BiLSTM (TPA-BiLSTM)
    - CNN-GRU混合模型

- **工具函数**：
    - ADF平稳性检验
    - 数据标准化/归一化
    - 滚动窗口预测
    - 时间序列可视化 (ACF, PACF, 残差图)

## 依赖环境

项目使用Python 3.x，需要以下包：

- numpy
- pandas
- scikit-learn
- statsmodels
- tensorflow/keras
- matplotlib
- seaborn