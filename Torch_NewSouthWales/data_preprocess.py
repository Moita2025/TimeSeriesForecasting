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

# 创建完整的 datetime 列
df_long['hour']   = df_long['time_str'].str.split(':', expand=True)[0].astype(int)
df_long['minute'] = df_long['time_str'].str.split(':', expand=True)[1].astype(int)
df_long['datetime'] = df_long['date'] + pd.to_timedelta(df_long['hour'], unit='h') + pd.to_timedelta(df_long['minute'], unit='min')

# 清理 & 排序
df_long = df_long.sort_values(['Customer', 'Consumption Category', 'datetime'])
df_long = df_long.drop(columns=['time_str', 'hour', 'minute'])

print("长表形状：", df_long.shape)

# ────────────────────────────────────────────────
# 3. 按 Customer + Category 补齐缺失时间点 & 插值
# ────────────────────────────────────────────────
print("\n开始补齐缺失半小时数据...")

result_list = []
warning_days = []

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

        # ─── 改进：更安全的日期范围 ───────────────────────────────
        # 只补到数据实际出现的最小/最大时间戳所在天的完整半小时
        min_dt = sub.index.min()
        max_dt = sub.index.max()

        start = min_dt.floor('D')                     # 当天 00:00
        end   = max_dt.normalize() + pd.Timedelta(hours=23, minutes=30)  # 当天 23:30

        full_index = pd.date_range(start, end, freq='30min')

        sub = sub.reindex(full_index)

        # 计算每天的缺失比例（用于后续分析）
        sub['is_missing'] = sub['kWh'].isna()
        daily_missing = sub.groupby(sub.index.date)['is_missing'].mean().rename('missing_ratio')

        # 记录严重缺失的日子（例如缺失 > 30%）
        bad_days = daily_missing[daily_missing > 0.3]
        if not bad_days.empty:
            warning_days.append((cust, cat, bad_days))

        # ─── 插值 ──────────────────────────────────────────────────
        # 先 clip 负值（以防万一）
        sub['kWh'] = sub['kWh'].clip(lower=0)

        # 线性插值（双向）
        sub['kWh'] = sub['kWh'].interpolate(method='linear', limit_direction='both')

        # 如果仍有 NaN（比如整段开头/结尾缺失），用前后最近值填充
        sub['kWh'] = sub['kWh'].ffill().bfill()

        # 固定属性填充
        for col in ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'Row Quality']:
            if col in sub.columns:
                sub[col] = sub[col].ffill().bfill()

        # 还原 date 列
        sub['date'] = sub.index.date

        # 加入缺失比例（按天）
        sub = sub.reset_index().rename(columns={'index': 'datetime'})
        sub = sub.merge(daily_missing, left_on='date', right_index=True, how='left')

        result_list.append(sub)

# 合并所有子表
df_long_filled = pd.concat(result_list, ignore_index=True)

print("补齐后长表形状：", df_long_filled.shape)

if warning_days:
    print("\n⚠️ 警告：以下客户-类别存在严重缺失日（缺失率>30%）：")
    for cust, cat, bad_days in warning_days[:5]:  # 只打印前5个，避免输出过多
        #print(f"  Customer {cust} | Category {cat} | {bad_days.index.tolist()} → 缺失率：{bad_days.values.round(3).tolist()}")
        print(f"  Customer {cust} | Category {cat} → 缺失天数：{len(bad_days.values.round(3).tolist())}")
    if len(warning_days) > 5:
        print(f"  ... 共 {len(warning_days)} 个客户-类别组合有严重缺失日（未全部列出）")

# ────────────────────────────────────────────────
# 4. 按日聚合 → 每日长表
# ────────────────────────────────────────────────
daily = df_long_filled.groupby(
    ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date', 'missing_ratio'],
    observed=True
)['kWh'].sum().reset_index(name='daily_total_kWh')

daily['daily_total_kWh'] = daily['daily_total_kWh'].round(3)
daily['missing_ratio'] = daily['missing_ratio'].round(3)

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

# 额外输出：缺失率分布概览
print("\n每日缺失率分布（按类别）：")
print(daily.groupby('Consumption Category')['missing_ratio'].describe().round(3))