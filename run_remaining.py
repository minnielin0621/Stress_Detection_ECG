from pathlib import Path
import subprocess
import sys

raw_dir = Path("data/raw/Raw ECG project")
output_dir = Path("data/cleaned/1000")

edf_files = sorted(raw_dir.glob("*_LAB_Conditions_ECG.edf"))

print(f"找到 {len(edf_files)} 個 EDF 檔案", flush=True)

failed = []

for index, edf_file in enumerate(edf_files, start=1):
    participant = edf_file.name.split("_")[0]
    output_file = output_dir / f"{participant}.parquet"

    if output_file.exists():
        print(f"[{index}/{len(edf_files)}] 跳過 {participant}（已完成）", flush=True)
        continue

    print(f"\n[{index}/{len(edf_files)}] 開始處理 {participant}", flush=True)

    command = [
        sys.executable,
        "preprocessing.py",
        "--participant_number", participant,
        "--data_chunk", "-1",
        "--sample_frequency", "1000",
        "--number_processors", "1",
    ]

    result = subprocess.run(command)

    if result.returncode == 0 and output_file.exists():
        print(f"完成：{participant}", flush=True)
    else:
        print(f"失敗：{participant}", flush=True)
        failed.append(participant)

print("\n===== 全部處理結束 =====")

if failed:
    print("失敗的受試者：", failed)
else:
    print("所有找到的 EDF 檔案均已有輸出檔案")
