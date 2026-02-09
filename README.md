# Time Series Prediction Models

A comprehensive collection of time series forecasting models implemented in both TensorFlow/Keras and PyTorch, targeting different datasets and scenarios (synthetic electrical load data and real-world solar home electricity data from New South Wales).

## Project Structure

### tf-syndt

- Implements univariate/multivariate time series models on synthetic electrical load data
- Includes statistical (ARIMA/ARIMAX) and deep learning models (BiLSTM, AM-BiLSTM, TPA-BiLSTM, CNN-GRU)
- Provides utility functions for time series preprocessing, stationarity testing, and visualization

### th-nsw-solar-home

- Implements advanced time series models on real solar home electricity data from New South Wales
- Built with PyTorch, covering diverse models (BiLSTM variants, CNN-GRU, DNN, iTransformer, tree-based models)
- Includes data preprocessing pipelines for real-world dataset handling

## Core Features

- Cross-framework implementation (TensorFlow/Keras + PyTorch)
- Support for both synthetic and real-world time series data
- Comprehensive model coverage:
    - Statistical models: ARIMA, ARIMAX
    - Deep learning models: BiLSTM, AM-BiLSTM, TPA-BiLSTM, CNN-GRU, DNN, iTransformer
    - Traditional ML models: RandomForest, XGBoost
- Complete data preprocessing, evaluation and visualization utilities

## Requirements

### tf-syndt

- Python 3.x
- numpy, pandas, scikit-learn, statsmodels, tensorflow/keras, matplotlib, seaborn

### th-nsw-solar-home

- Python 3.x
- numpy, pandas, scikit-learn, torch, matplotlib, seaborn, xgboost, scikit-learn

---

# 时间序列预测模型

本项目是一个全面的时间序列预测模型合集，基于 TensorFlow/Keras 和 PyTorch 双框架实现，面向不同数据集和场景（合成电力负荷数据、新南威尔士州真实居民光伏用电数据）。

## 项目结构

### tf-syndt

- 基于合成电力负荷数据实现单/多变量时间序列模型
- 包含统计模型（ARIMA/ARIMAX）和深度学习模型（BiLSTM、AM-BiLSTM、TPA-BiLSTM、CNN-GRU）
- 提供时间序列预处理、平稳性检验、可视化等工具函数

### th-nsw-solar-home

- 基于新南威尔士州真实居民光伏用电数据实现进阶时间序列模型
- 基于 PyTorch 构建，覆盖多类模型（BiLSTM 变体、CNN-GRU、DNN、iTransformer、树基模型）
- 包含面向真实世界数据集的预处理流程

## 核心特性

- 跨框架实现（TensorFlow/Keras + PyTorch）
- 支持合成数据和真实世界时间序列数据
- 全面的模型覆盖：
    - 统计模型：ARIMA、ARIMAX
    - 深度学习模型：BiLSTM、AM-BiLSTM、TPA-BiLSTM、CNN-GRU、DNN、iTransformer
    - 传统机器学习模型：RandomForest、XGBoost
- 完整的数据预处理、评估和可视化工具
- 适配不同数据源的工程化实现

## 依赖环境

### tf-syndt

- Python 3.x
- numpy, pandas, scikit-learn, statsmodels, tensorflow/keras, matplotlib, seaborn

### th-nsw-solar-home

- Python 3.x
- numpy, pandas, scikit-learn, torch, matplotlib, seaborn, xgboost, scikit-learn
