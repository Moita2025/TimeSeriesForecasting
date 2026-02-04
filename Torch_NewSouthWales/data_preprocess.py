import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm.auto import tqdm

# ────────────────────────────────────────────────
# 1. 读取原始数据
# ────────────────────────────────────────────────
file_path = '2012-2013 Solar home electricity data v2.csv'

df = pd.read_csv(
    file_path,
    skiprows=1,                      # 已验证可解决问题
    dtype={
        'Customer': 'int32',
        'Generator Capacity': 'float32',
        'Postcode': 'int32',
        'Consumption Category': 'category',
    },
    parse_dates=['date'],
    dayfirst=True,
    date_format='%d/%m/%Y',
    encoding='utf-8',                # 如有问题可改为 'utf-8-sig'
    low_memory=False
)

print("读取完成，形状：", df.shape)
print("date 类型：", df['date'].dtype)

# ────────────────────────────────────────────────
# 2. 转换为半小时长表
# ────────────────────────────────────────────────
id_cols = ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date', 'Row Quality']

# 只选真正的时间列（包含 : 的列）
time_cols = [col for col in df.columns if ':' in col]

df_long = pd.melt(
    df,
    id_vars=id_cols,
    value_vars=time_cols,
    var_name='time_str',
    value_name='kWh'
)

# 创建完整的 datetime 列（date + time）
df_long['hour']   = df_long['time_str'].str.split(':', expand=True)[0].astype(int)
df_long['minute'] = df_long['time_str'].str.split(':', expand=True)[1].astype(int)
df_long['datetime'] = df_long['date'] + pd.to_timedelta(df_long['hour'], unit='h') + pd.to_timedelta(df_long['minute'], unit='min')

# 排序 & 清理临时列
df_long = df_long.sort_values(['Customer', 'Consumption Category', 'datetime'])
df_long = df_long.drop(columns=['time_str', 'hour', 'minute'])

print("长表形状：", df_long.shape)

# ────────────────────────────────────────────────
# 3. 按 Customer + Category 补齐缺失时间点 & 插值
# ────────────────────────────────────────────────
print("\n开始补齐缺失半小时数据...")

result_list = []

customers = df_long['Customer'].unique()
categories = df_long['Consumption Category'].cat.categories  # ['CL','GC','GG']

for cust in tqdm(customers, desc="Customers"):
    for cat in categories:
        sub = df_long[
            (df_long['Customer'] == cust) &
            (df_long['Consumption Category'] == cat)
        ].copy()

        if sub.empty:
            continue

        sub = sub.set_index('datetime')

        # 理论完整时间范围（30分钟粒度）
        start = sub.index.min().floor('D')          # 当天 00:00
        end   = sub.index.max().ceil('D') - timedelta(seconds=1)  # 当天 23:59:59
        full_index = pd.date_range(start, end, freq='30min')

        sub = sub.reindex(full_index)

        # 插值 consumption (kWh)
        sub['kWh'] = sub['kWh'].interpolate(method='linear', limit_direction='both')

        # 固定属性向前/向后填充（理论上不应该有缺，但以防万一）
        for col in ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'Row Quality']:
            if col in sub.columns:
                sub[col] = sub[col].ffill().bfill()

        # 还原 date 列（用于后续按日聚合）
        sub['date'] = sub.index.date

        result_list.append(sub.reset_index().rename(columns={'index': 'datetime'}))

# 合并所有补齐后的子表
df_long_filled = pd.concat(result_list, ignore_index=True)

print("补齐后长表形状：", df_long_filled.shape)

# ────────────────────────────────────────────────
# 4. 按日聚合 → 每日长表
# ────────────────────────────────────────────────
daily = df_long_filled.groupby(
    ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date'],
    observed=True
)['kWh'].sum().reset_index(name='daily_total_kWh')

daily['daily_total_kWh'] = daily['daily_total_kWh'].round(3)

# 排序
daily = daily.sort_values(['Customer', 'date', 'Consumption Category'])

# ────────────────────────────────────────────────
# 5. 保存 & 预览
# ────────────────────────────────────────────────
daily.to_csv('data_daily_long.csv', index=False, float_format='%.3f')

print("\n处理完成，已保存为 data_daily_long.csv")
print("每日数据形状：", daily.shape)
print("\n前9行预览：")
print(daily.head(9))