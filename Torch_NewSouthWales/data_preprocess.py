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
    skiprows=1,
    dtype={
        'Customer': 'int32',
        'Generator Capacity': 'float32',
        'Postcode': 'int32',
        'Consumption Category': 'category',
    },
    parse_dates=['date'],
    dayfirst=True,
    date_format='%d/%m/%Y',
    encoding='utf-8',
    low_memory=False
)

print("读取完成，形状：", df.shape)

# ────────────────────────────────────────────────
# 2. 转换为半小时长表
# ────────────────────────────────────────────────
id_cols = ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date', 'Row Quality']
time_cols = [col for col in df.columns if ':' in col]

df_long = pd.melt(
    df, id_vars=id_cols, value_vars=time_cols,
    var_name='time_str', value_name='kWh'
)

df_long['hour']   = df_long['time_str'].str.split(':', expand=True)[0].astype(int)
df_long['minute'] = df_long['time_str'].str.split(':', expand=True)[1].astype(int)
df_long['datetime'] = df_long['date'] + pd.to_timedelta(df_long['hour'], unit='h') + pd.to_timedelta(df_long['minute'], unit='min')

df_long = df_long.sort_values(['Customer', 'Consumption Category', 'datetime'])
df_long = df_long.drop(columns=['time_str', 'hour', 'minute'])

print("长表形状：", df_long.shape)

# ────────────────────────────────────────────────
# 3. 按 Customer + Category 补齐 & 插值（新增严重缺失日处理）
# ────────────────────────────────────────────────
print("\n开始补齐缺失半小时数据...")

result_list = []
warning_days = []

customers = df_long['Customer'].unique()
categories = df_long['Consumption Category'].cat.categories

for cust in tqdm(customers, desc="Customers"):
    for cat in categories:
        sub = df_long[
            (df_long['Customer'] == cust) &
            (df_long['Consumption Category'] == cat)
        ].copy()

        if sub.empty:
            continue

        sub = sub.set_index('datetime')

        # 安全日期范围
        min_dt = sub.index.min()
        max_dt = sub.index.max()
        start = min_dt.floor('D')
        end   = max_dt.normalize() + pd.Timedelta(hours=23, minutes=30)
        full_index = pd.date_range(start, end, freq='30min')

        sub = sub.reindex(full_index)

        # 缺失统计
        sub['is_missing'] = sub['kWh'].isna()
        daily_missing = sub.groupby(sub.index.date)['is_missing'].mean().rename('missing_ratio')
        bad_days = daily_missing[daily_missing > 0.3]
        if not bad_days.empty:
            warning_days.append((cust, cat, bad_days))

        # clip 负值
        sub['kWh'] = sub['kWh'].clip(lower=0)

        # 线性插值（短缺失）
        sub['kWh'] = sub['kWh'].interpolate(method='linear', limit_direction='both')
        sub['kWh'] = sub['kWh'].ffill().bfill()

        # ─── 新增：严重缺失日使用 weekday + time-of-day 历史均值插补 ───
        if not bad_days.empty:
            # 计算 good days 的典型 profile
            bad_dates = bad_days.index
            good_mask = ~sub.index.to_series().dt.date.isin(bad_dates)
            good_sub = sub[good_mask].copy()
            good_sub['weekday'] = good_sub.index.weekday
            good_sub['tod']     = good_sub.index.time

            typical_profile = good_sub.groupby(['weekday', 'tod'])['kWh'].mean() \
                                      .reset_index(name='typical_kWh')

            # 对 bad days 进行替换
            bad_mask = sub.index.to_series().dt.date.isin(bad_dates)
            bad_sub = sub[bad_mask].copy()
            bad_sub['weekday'] = bad_sub.index.weekday
            bad_sub['tod']     = bad_sub.index.time

            bad_sub = bad_sub.merge(typical_profile, on=['weekday', 'tod'], how='left')
            bad_sub['kWh'] = bad_sub['typical_kWh'].fillna(0)   # 极少数无参考值 → 0

            sub.loc[bad_mask, 'kWh'] = bad_sub['kWh'].values

            # 该天缺失率标记为 0（已合理填充）
            daily_missing.loc[bad_dates] = 0.0

        # 固定属性填充
        for col in ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'Row Quality']:
            if col in sub.columns:
                sub[col] = sub[col].ffill().bfill()

        # 还原 date
        sub['date'] = sub.index.date

        # 加入缺失比例（已更新）
        sub = sub.reset_index().rename(columns={'index': 'datetime'})
        sub = sub.merge(daily_missing, left_on='date', right_index=True, how='left')

        result_list.append(sub)

# 合并
df_long_filled = pd.concat(result_list, ignore_index=True)
print("补齐后长表形状：", df_long_filled.shape)

if warning_days:
    print("\n⚠️ 以下客户-类别存在严重缺失日（>30%），已使用同 weekday 同时间槽历史均值插补：")
    for cust, cat, bad_days in warning_days[:10]:
        print(f"  Customer {cust} | Category {cat} → {len(bad_days)} 天")
    if len(warning_days) > 10:
        print(f"  ... 共 {len(warning_days)} 个客户-类别组合")

# ────────────────────────────────────────────────
# 4. 按日聚合
# ────────────────────────────────────────────────
daily = df_long_filled.groupby(
    ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date', 'missing_ratio'],
    observed=True
)['kWh'].sum().reset_index(name='daily_total_kWh')

daily['daily_total_kWh'] = daily['daily_total_kWh'].round(3)
daily['missing_ratio']   = daily['missing_ratio'].round(3)

daily = daily.sort_values(['Customer', 'date', 'Consumption Category'])

# ────────────────────────────────────────────────
# 5. 保存 & 预览
# ────────────────────────────────────────────────
daily.to_csv('data_daily_long.csv', index=False, float_format='%.3f')

print("\n处理完成，已保存为 data_daily_long.csv")
print("每日数据形状：", daily.shape)
print("\n每日缺失率分布（按类别，最终）：")
print(daily.groupby('Consumption Category')['missing_ratio'].describe().round(3))