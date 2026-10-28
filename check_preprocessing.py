
import pandas as pd
from pathlib import Path

# ==============================
# Step 1: 讀取處理後的資料
# ==============================

file_path = Path("data/cleaned/1000/30100.parquet")

if not file_path.exists():
    raise FileNotFoundError(f"找不到檔案: {file_path}")

df = pd.read_parquet(file_path)

# ==============================
# Step 2: 基本資料資訊
# ==============================

print("\n===== 基本資訊 =====")
print("資料形狀 (rows, columns):", df.shape)
print("欄位名稱:")
print(df.columns.tolist())

print("\n前 5 筆資料:")
print(df.head().to_string())

# ==============================
# Step 3: 檢查缺失值
# ==============================

print("\n===== 缺失值 =====")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("沒有缺失值")
else:
    print(missing.head(20).to_string())

# ==============================
# Step 4: 檢查標籤分布
# ==============================

print("\n===== 標籤分布 =====")

for col in ["label", "category"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

# ==============================
# Step 5: 檢查 ECG 相關欄位
# ==============================

print("\n===== ECG 相關欄位 =====")

ecg_cols = [
    col for col in df.columns
    if "ECG" in col.upper() or "QUALITY" in col.upper()
]

print(ecg_cols)

for col in ecg_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"\n{col}:")
        print(df[col].describe().to_string())

# ==============================
# Step 6: 檢查訊號品質
# ==============================

print("\n===== 訊號品質 =====")

if "ECG_Quality" in df.columns:
    quality = pd.to_numeric(df["ECG_Quality"], errors="coerce")

    print(quality.describe())
    print("品質低於或等於 0.25 的筆數:",
          (quality <= 0.25).sum())
else:
    print("沒有找到 ECG_Quality 欄位")

print("\n===== 檢查完成 =====")
