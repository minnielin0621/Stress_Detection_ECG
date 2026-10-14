
# -*- coding: utf-8 -*-
"""
ReadData.py

功能：
1. 找出所有原始 ECG 的 EDF 檔案
2. 讀取實驗時間標記資料
3. 逐位受試者讀取 ECG
4. 建立每個 ECG 採樣點的時間
5. 找出該受試者對應的實驗時間紀錄

注意：
目前程式沒有真正切割 ECG、沒有指派 0/1 標籤，
也沒有輸出 CSV 或其他處理後的資料檔案。
"""

# ==============================
# Step 1：匯入需要的套件
# ==============================

import os             # 處理資料夾與檔案路徑
import pandas as pd    # 讀取與處理表格資料
import mne             # 讀取 EDF 格式的 ECG
import datetime        # 處理日期與時間


# ==============================
# Step 2：設定原始資料的位置
# ==============================

from utils.helper_path import RAW_DATA_PATH

folderpath = RAW_DATA_PATH

# 找出資料夾內所有檔名以 _ECG.edf 結尾的檔案
# filepaths：完整路徑，例如 D:\...\30225_LAB_Conditions_ECG.edf
filepaths = [
    os.path.join(folderpath, f)
    for f in os.listdir(folderpath)
    if f.endswith('_ECG.edf')
]

# files：只有檔案名稱，不包含資料夾路徑
files = [
    f for f in os.listdir(folderpath)
    if f.endswith('_ECG.edf')
]

# 取檔案名稱前 5 個字元作為受試者 ID
# 例如：30225_LAB_Conditions_ECG.edf → 30225
participants = [f[:5] for f in files]


# ==============================
# Step 3：讀取實驗時間標記檔
# ==============================

timestamps = pd.read_csv(
    os.path.join(folderpath, 'Timestamps_Merged.txt'),
    sep="\t",                  # 欄位用 Tab 分隔
    decimal=".",              # 小數點使用 .
    dtype={"Subject_ID": str} # 受試者 ID 保留字串格式
)

# 將實驗區間的開始時間轉成 datetime
# utc=True：將輸入時間視為 UTC
timestamps['LabelStart'] = pd.to_datetime(
    timestamps['LabelStart'],
    format="%Y-%m-%d %H:%M:%S",
    utc=True
)

# 將實驗區間的結束時間轉成 datetime
timestamps['LabelEnd'] = pd.to_datetime(
    timestamps['LabelEnd'],
    format="%Y-%m-%d %H:%M:%S",
    utc=True
)

# 確保受試者 ID 都是字串
timestamps["Subject_ID"] = timestamps["Subject_ID"].astype(str)


# ==============================
# Step 4：設定 ECG 取樣率
# ==============================

fs = 1000

# 代表假設 ECG 每秒有 1000 個採樣點
# 注意：這只是設定變數，不是降採樣
# 也尚未驗證 EDF 真正的取樣率


# ==============================
# Step 5：逐位受試者讀取 ECG
# ==============================

for p in range(len(participants)):

    # 印出目前正在處理的受試者
    print(participants[p])

    # 印出目前進度，例如 1 / 127
    print(p + 1, '/', len(participants))

    # 讀取受試者的 EDF 檔案
    # preload=True：將訊號載入記憶體
    ECG_edf = mne.io.read_raw_edf(
        filepaths[p],
        preload=True,
        verbose=False
    )

    # 取出 EDF 的第一個訊號通道
    # 注意：必須確認第一個通道確實是 ECG
    ECG = ECG_edf.get_data()[0]


    # ==============================
    # Step 6：建立 ECG 時間軸
    # ==============================

    # 取得 EDF annotation 記錄的起始時間
    start_time = ECG_edf.annotations.orig_time

    # 計算最後一個採樣點的時間
    end_time = start_time + datetime.timedelta(
        seconds=(len(ECG) - 1) * (1 / fs)
    )

    # 將每個採樣點的相對秒數轉成實際時間
    time_vector = [
        start_time + datetime.timedelta(seconds=delta)
        for delta in ECG_edf.times
    ]

    # 將時間轉成文字格式
    # 例如 2026-01-01 10:00:00.001000
    time_vector = [
        t.strftime("%Y-%m-%d %H:%M:%S.%f")
        for t in time_vector
    ]


    # ==============================
    # Step 7：取得該受試者的實驗紀錄
    # ==============================

    timestamps_temp = timestamps.loc[
        timestamps['Subject_ID'] == participants[p], :
    ]
    print("Participant:", participants[p])
    print("Channels:", ECG_edf.ch_names)
    print("Sampling rate:", ECG_edf.info["sfreq"])
    print("ECG samples:", len(ECG))
    print("Timestamp records:", len(timestamps_temp))
    print("------------------------------")

    # 到這裡程式就結束了！
    #
    # 目前沒有：
    # 1. 使用 LabelStart / LabelEnd 切割 ECG
    # 2. 將實驗條件轉成 Stress / Non-stress
    # 3. 將標籤與 ECG 採樣點合併
    # 4. 使用 to_csv() 或其他方法儲存結果
