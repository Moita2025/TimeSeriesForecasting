# Time Series Forecasting for Solar Home Electricity (New South Wales)

A PyTorch-based implementation of time series forecasting models for solar home electricity consumption/generation data from New South Wales, Australia. The dataset contains real-world residential electricity usage and solar generation data, enabling the evaluation of diverse forecasting models on practical time series scenarios.

## Dataset Information

### Source

The dataset is sourced from the New South Wales Government Open Data Portal:  
https://data.nsw.gov.au/data/dataset/solar-home-electricty-data  

### License

This dataset is licensed under the Creative Commons Attribution 4.0 International (CCA 4.0) license.  
**Attribution**: State of New South Wales. For current information go to www.nsw.gov.au.

### Data Description

- Contains daily solar home electricity consumption and generation data for residential households in New South Wales (2012-2013)
- Raw data is provided in zip format, with preprocessed daily-long format CSV for direct model training
- Covers multiple households with temporal electricity usage/solar generation metrics

## Implemented Models

### Deep Learning Models

- BiLSTM (Bidirectional Long Short-Term Memory)
- AM-BiLSTM (Attention Mechanism BiLSTM)
- TPA-BiLSTM (Temporal Pattern Attention BiLSTM)
- CNN-GRU (Convolutional Neural Network + Gated Recurrent Unit)
- DNN (Deep Neural Network)
- iTransformer (Improved Transformer for Time Series)

### Traditional Machine Learning Models

- RandomForest (Random Forest Regressor)
- XGBoost (Extreme Gradient Boosting)

## Core Utilities

- Data preprocessing pipeline for real-world time series data (handling missing values, normalization, feature engineering)
- Model training/evaluation utilities for non-tree-based models
- Dataset loading and transformation functions for PyTorch

## Requirements

The project uses Python 3.x and requires the following packages:

- numpy
- pandas
- scikit-learn
- torch
- matplotlib
- seaborn
- xgboost

---

# 新南威尔士州光伏住宅用电时间序列预测

基于 PyTorch 实现的时间序列预测模型，针对澳大利亚新南威尔士州（NSW）真实居民光伏住宅用电数据进行负荷/发电量预测。该数据集包含真实住宅的用电和光伏发电记录，可用于验证各类预测模型在实际时间序列场景中的性能。

## 数据集信息

### 数据源

数据集来源于新南威尔士州政府开放数据门户：  
https://data.nsw.gov.au/data/dataset/solar-home-electricty-data  

### 授权协议

本数据集采用知识共享署名 4.0 国际许可协议（CCA 4.0）。 

**引用标注**：State of New South Wales. For current information go to www.nsw.gov.au.

### 数据说明

- 包含新南威尔士州居民住宅 2012-2013 年的日度光伏发电/用电数据
- 原始数据为 zip 压缩包格式，提供预处理后的长格式日度 CSV 数据文件
- 覆盖多户家庭的用电/发电时序指标，包含丰富的真实世界数据特征

## 已实现模型

### 深度学习模型

- BiLSTM（双向长短期记忆网络）
- AM-BiLSTM（注意力机制双向长短期记忆网络）
- TPA-BiLSTM（时间模式注意力双向长短期记忆网络）
- CNN-GRU（卷积神经网络+门控循环单元）
- DNN（深度神经网络）
- iTransformer（改进型时间序列Transformer）

### 传统机器学习模型

- RandomForest（随机森林回归）
- XGBoost（极端梯度提升）

## 核心工具

- 面向真实世界时间序列的数据预处理流程（缺失值处理、标准化、特征工程）
- 非树基模型的训练/评估工具函数
- 适用于 PyTorch 的数据集加载与转换函数

## 依赖环境

项目使用 Python 3.x，需要以下包：
- numpy
- pandas
- scikit-learn
- torch
- matplotlib
- seaborn
- xgboost
