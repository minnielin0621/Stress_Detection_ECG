import pandas as pd
import numpy as np

# 讀取前處理後的 ECG
path = "data/cleaned/1000/30100.parquet"
df = pd.read_parquet(path)

# 時間轉換
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 計算相鄰採樣點的時間差
dt = df["timestamp"].diff().dt.total_seconds()

# 找出大於 10 ms 的缺口
gap_indices = np.where(dt > 0.01)[0]

print("========== ECG 時間缺口 ==========")
print("缺口數量:", len(gap_indices))

for idx in gap_indices:
    print(
        f"位置: {idx}, "
        f"缺口: {dt.iloc[idx]:.3f} 秒, "
        f"時間: {df['timestamp'].iloc[idx]}"
    )

# 120 秒視窗，10 秒滑動
window_seconds = 120
step_seconds = 10

# 使用真實 timestamp 建立視窗
start_time = df["timestamp"].iloc[0]
end_time = df["timestamp"].iloc[-1]

gap_times = df["timestamp"].iloc[gap_indices]

affected = []
total = 0

current_start = start_time

while current_start + pd.Timedelta(seconds=window_seconds) <= end_time:

    current_end = current_start + pd.Timedelta(
        seconds=window_seconds
    )

    # 檢查視窗內是否包含缺口
    has_gap = (
        (gap_times >= current_start) &
        (gap_times < current_end)
    ).any()

    if has_gap:
        affected.append((current_start, current_end))

    total += 1

    current_start += pd.Timedelta(seconds=step_seconds)

print("\n========== 120 秒視窗 ==========")
print("時間軸上的視窗總數:", total)
print("受到缺口影響的視窗:", len(affected))
print("未受到缺口影響的視窗:", total - len(affected))

print("\n========== 受影響視窗 ==========")

for start, end in affected:
    print(start, "→", end)