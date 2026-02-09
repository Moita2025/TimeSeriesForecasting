import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 配置 =====================
DATA_DIR = Path("data_csvs")                     # 你的文件夹名
OUTPUT_DIR = Path("data_processed")
OUTPUT_DIR.mkdir(exist_ok=True)

HALF_HOUR_FILE = OUTPUT_DIR / "nsw_halfhour_201207_201306.csv"
DAILY_FILE = OUTPUT_DIR / "nsw_daily_201207_201306.csv"

# ===================== 1. 读取并合并所有月份文件 =====================
print("正在读取所有月份文件...")

all_dfs = []
for csv_path in tqdm(sorted(DATA_DIR.glob("PRICE_AND_DEMAND_*.csv"))):
    month_df = pd.read_csv(
        csv_path,
        parse_dates=["SETTLEMENTDATE"],
        dayfirst=False,           # 格式是 2012/07/01 00:30:00
        date_format="%Y/%m/%d %H:%M:%S",
    )
    all_dfs.append(month_df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"合并完成，总行数：{len(df):,}")

# 排序 & 去重（以防万一）
df = df.sort_values("SETTLEMENTDATE").drop_duplicates(subset=["SETTLEMENTDATE"]).reset_index(drop=True)

# ===================== 2. 基本清洗 =====================
print("\n基本信息与清洗：")

# 检查 REGION 是否都是 NSW1
print("REGION 唯一值：", df["REGION"].unique())

if len(df["REGION"].unique()) == 1:
    df = df.drop(columns=["REGION"])
    print("已删除 REGION 列。")

# 检查 PERIODTYPE 分布（正常应大部分是 TRADE）
print("\nPERIODTYPE 分布：")
print(df["PERIODTYPE"].value_counts(dropna=False))

periodtype_counts = df["PERIODTYPE"].value_counts(dropna=False)
if len(periodtype_counts) == 1:
    df = df.drop(columns=["PERIODTYPE"])
    print("已删除 PERIODTYPE 列。")

# 缺失值检查
print("\n缺失值统计：")
print(df.isna().sum())

# 负值检查（负荷和价格理论上不应为负，但有时有异常）
print(f"\nTOTALDEMAND < 0 的记录数：{ (df['TOTALDEMAND'] < 0).sum() }")
print(f"RRP < 0 的记录数：{ (df['RRP'] < -100).sum() }  (极端负价)")

# 可选：把极端负价替换为 NaN 或前值（视需求）
# df.loc[df['RRP'] < -100, 'RRP'] = np.nan
# df['RRP'] = df['RRP'].fillna(method='ffill')

# ===================== 3. 时间特征（可选，为后续建模准备） =====================
df["date"] = df["SETTLEMENTDATE"].dt.date
df["year"] = df["SETTLEMENTDATE"].dt.year
df["month"] = df["SETTLEMENTDATE"].dt.month
df["day"] = df["SETTLEMENTDATE"].dt.day
df["hour"] = df["SETTLEMENTDATE"].dt.hour
df["minute"] = df["SETTLEMENTDATE"].dt.minute
df["weekday"] = df["SETTLEMENTDATE"].dt.weekday
df["is_weekend"] = df["weekday"] >= 5

# ===================== 4. 保存半小时级数据 =====================
print(f"\n保存半小时级数据 → {HALF_HOUR_FILE}")
df.to_csv(HALF_HOUR_FILE, index=False, float_format="%.3f")
print("保存完成")

# ===================== 5. （可选）按天聚合 =====================
print("\n生成日级别聚合...")

daily = df.groupby("date").agg({
    "TOTALDEMAND": ["mean", "max", "min", "sum"],
    "RRP": ["mean", "max", "min", "median"],
    "SETTLEMENTDATE": "count"   # 检查每天是否48条
}).reset_index()

daily.columns = [
    "date",
    "demand_mean", "demand_max", "demand_min", "demand_sum",
    "rrp_mean", "rrp_max", "rrp_min", "rrp_median",
    "record_count"
]

# 检查是否每天都是48条记录
print("\n每天记录数分布：")
print(daily["record_count"].value_counts().sort_index())

daily["date"] = pd.to_datetime(daily["date"])

print(f"保存日级别数据 → {DAILY_FILE}")
daily.to_csv(DAILY_FILE, index=False, float_format="%.3f")
print("日级别聚合完成")

print("\n预处理结束。")
print(f"半小时级文件：{HALF_HOUR_FILE}")
print(f"日级别文件  ：{DAILY_FILE}")