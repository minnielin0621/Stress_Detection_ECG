"""Participant-wise ECG preprocessing without ECG quality row filtering.

Default: find all EDFs recursively, skip completed Parquet files, inspect all.
Outputs only to data/cleaned_no_quality_filter/<sampling_rate>/.
"""

import argparse
import os
import re
import time
import traceback
from pathlib import Path

import pandas as pd

from sia import Preprocessing
from sia.io import Metadata, read_edf, write_csv
from sia.preprocessors import neurokit
from sia.encoders import GroupEncoder
from utils.helper_path import RAW_DATA_PATH, CLEANED_DATA_PATH
from utils.helper_functions import create_directory

EDF_PATTERN = re.compile(r"^(\d{5})_LAB_Conditions_ECG\.edf$", re.IGNORECASE)
QUALITY_THRESHOLD = 0.25
GAP_THRESHOLD_MS = 10.0

CATEGORY_GROUPS = {
    "baseline": ["Sitting", "Recov1", "Recov2", "Recov3", "Recov4", "Recov5", "Recov6"],
    "mental_stress": ["TA", "SSST_Sing_countdown", "Pasat", "Raven", "TA_repeat", "Pasat_repeat"],
    "high_physical_activity": ["Treadmill1", "Treadmill2", "Treadmill3", "Treadmill4",
                               "Walking_fast_pace", "Cycling", "stairs_up_and_down"],
    "moderate_physical_activity": ["Walking_own_pace", "Dishes", "Vacuum"],
    "low_physical_activity": ["Standing", "Lying_supine", "Recov_standing"],
}


def locate_edfs(raw_root, sample_frequency, data_chunk, participant_number):
    root = Path(raw_root)
    if data_chunk == -1:
        search_root = root
    elif data_chunk in (1, 2, 3, 4, 5):
        search_root = root / str(sample_frequency) / f"part_{data_chunk}"
    else:
        raise ValueError("--data_chunk must be -1 or 1 through 5")

    if not search_root.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {search_root}\n"
            "Your EDFs may be directly under Raw ECG project. Use --data_chunk -1."
        )

    found = {}
    for path in sorted(search_root.rglob("*")):
        if not path.is_file():
            continue
        match = EDF_PATTERN.fullmatch(path.name)
        if not match:
            continue
        participant_id = match.group(1)
        if participant_number != -1 and participant_id != str(participant_number):
            continue
        if participant_id in found:
            raise ValueError(f"Duplicate participant {participant_id}: {found[participant_id]} and {path}")
        found[participant_id] = path
    return sorted(found.items())


def validate_existing_parquet(path, participant_id):
    """Conservative structural check; cannot prove provenance of existing data."""
    if not path.exists():
        return False, "File does not exist"
    try:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema.names)
        required = {"timestamp", "ECG_Clean", "ECG_R_Peaks", "ECG_Quality", "label", "category"}
        missing = sorted(required - names)
        if missing:
            return False, f"Missing columns: {missing}"
        if parquet.metadata.num_rows <= 0:
            return False, "Parquet has zero rows"
        if path.stem != participant_id:
            return False, "Participant filename mismatch"
        return True, f"{parquet.metadata.num_rows:,} rows, required columns present"
    except Exception as exc:
        return False, f"Cannot read Parquet metadata: {exc}"


def process_one(edf_path, output_dir, metadata_path, args):
    Preprocessing(num_proc=args.number_processors) \
        .data(read_edf(str(edf_path),
                       Metadata(str(metadata_path)).on_regex(r"[0-9]{5}"),
                       sampling_rate=args.sample_frequency)) \
        .rename({"category": "label"}) \
        .encode({"label": "category"}, GroupEncoder(CATEGORY_GROUPS)) \
        .filter(lambda category: [value is not None for value in category]) \
        .process(neurokit(sampling_rate=args.sample_frequency, method=args.method)) \
        .filter(lambda label: [value != "Lying_supine" for value in label]) \
        .to(write_csv(str(output_dir / "[0-9]{5}.csv."), use_parquet=True))
    # Deliberately no ECG_Quality > 0.25 filter.


def inspect_one(path, sampling_rate):
    """Inspect an existing or newly generated Parquet; return summary and gap rows."""
    df = pd.read_parquet(path)
    participant = path.stem
    print(f"\n===== INSPECTION: {participant} =====", flush=True)
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("First five rows:\n", df.head().to_string(index=False))

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("Missing values:\n", missing.head(20).to_string() if not missing.empty else "None")
    for name in ("label", "category"):
        if name in df:
            print(f"{name} distribution:\n{df[name].value_counts(dropna=False).to_string()}")

    ecg_cols = [name for name in df.columns if "ECG" in name.upper() or "QUALITY" in name.upper()]
    print("ECG-related columns:", ecg_cols)
    for name in ecg_cols:
        if pd.api.types.is_numeric_dtype(df[name]):
            print(f"{name}:\n{df[name].describe().to_string()}")

    quality_count = None
    quality_pct = None
    if "ECG_Quality" in df:
        quality = pd.to_numeric(df["ECG_Quality"], errors="coerce")
        quality_count = int((quality <= QUALITY_THRESHOLD).sum())
        valid = int(quality.notna().sum())
        quality_pct = round(quality_count / valid * 100, 4) if valid else None
        print(f"Low-quality samples (<= {QUALITY_THRESHOLD}): {quality_count:,} ({quality_pct}%)")

    gap_rows = []
    negative_or_zero_diffs = None
    invalid_timestamps = None
    if "timestamp" in df:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_timestamps = int(ts.isna().sum())
        diff = ts.diff().dt.total_seconds()
        negative_or_zero_diffs = int((diff <= 0).sum())
        positions = (diff > GAP_THRESHOLD_MS / 1000).to_numpy().nonzero()[0]
        for pos in positions:
            record = {
                "participant": participant,
                "row_before": int(pos - 1),
                "row_after": int(pos),
                "time_before": ts.iloc[pos - 1],
                "time_after": ts.iloc[pos],
                "gap_seconds": float(diff.iloc[pos]),
                "label_before": df["label"].iloc[pos - 1] if "label" in df else None,
                "label_after": df["label"].iloc[pos] if "label" in df else None,
                "category_before": df["category"].iloc[pos - 1] if "category" in df else None,
                "category_after": df["category"].iloc[pos] if "category" in df else None,
            }
            gap_rows.append(record)
            print(f"GAP {ts.iloc[pos-1]} -> {ts.iloc[pos]} ({diff.iloc[pos]:.3f} s)")
        print(f"Gaps > {GAP_THRESHOLD_MS:g} ms: {len(gap_rows)}")
        print(f"Non-increasing timestamp differences: {negative_or_zero_diffs}")
        print(f"Invalid timestamps: {invalid_timestamps}")
    else:
        print("WARNING: timestamp column not found")

    summary = {
        "participant": participant,
        "samples": len(df),
        "columns": len(df.columns),
        "low_quality_samples": quality_count,
        "low_quality_percentage": quality_pct,
        "timestamp_gaps": len(gap_rows),
        "max_gap_seconds": max((x["gap_seconds"] for x in gap_rows), default=0.0),
        "non_increasing_timestamps": negative_or_zero_diffs,
        "invalid_timestamps": invalid_timestamps,
    }
    del df
    return summary, gap_rows


def save_summaries(output_dir, processing, inspection, gaps):
    # Save after each participant so Ctrl+C does not lose previous summaries.
    pd.DataFrame(processing, columns=[
        "participant", "status", "processing_minutes", "output_file", "error"
    ]).to_csv(output_dir / "processing_summary.csv", index=False)
    pd.DataFrame(inspection, columns=[
        "participant", "samples", "columns", "low_quality_samples", "low_quality_percentage",
        "timestamp_gaps", "max_gap_seconds", "non_increasing_timestamps", "invalid_timestamps"
    ]).to_csv(output_dir / "inspection_summary.csv", index=False)
    pd.DataFrame(gaps, columns=[
        "participant", "row_before", "row_after", "time_before", "time_after", "gap_seconds",
        "label_before", "label_after", "category_before", "category_after"
    ]).to_csv(output_dir / "timestamp_gap_details.csv", index=False)


def main(args):
    started = time.time()
    raw_root = Path(RAW_DATA_PATH)
    cleaned_parent = Path(CLEANED_DATA_PATH).parent
    output_dir = cleaned_parent / "cleaned_no_quality_filter" / str(args.sample_frequency)
    metadata_path = raw_root / "TimeStamps_Merged.txt"
    create_directory(str(output_dir))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    participants = locate_edfs(raw_root, args.sample_frequency, args.data_chunk, args.participant_number)
    print("\n" + "=" * 60)
    print("PREPROCESSING WITHOUT ECG QUALITY FILTERING")
    print("=" * 60)
    print(f"Raw EDF root: {raw_root}")
    print(f"Output directory: {output_dir}")
    print(f"EDF files found: {len(participants)}")
    print("ECG Quality Filter: DISABLED")
    print("Existing valid Parquet: SKIP PREPROCESSING, INSPECT")
    print("Existing invalid Parquet: DO NOT OVERWRITE")
    print("=" * 60)
    if not participants:
        print("No matching EDF files found. Check --data_chunk and raw directory.")
        return

    processing, inspection, gaps = [], [], []
    counts = {"processed": 0, "skipped": 0, "failed": 0, "invalid_existing": 0, "inspection_failed": 0}
    for index, (participant_id, edf_path) in enumerate(participants, 1):
        print(f"\n[{index}/{len(participants)}] Participant: {participant_id}")
        print(f"EDF: {edf_path}", flush=True)
        output_file = output_dir / f"{participant_id}.parquet"
        elapsed = 0.0
        status = None
        error = ""

        if output_file.exists():
            valid, reason = validate_existing_parquet(output_file, participant_id)
            if valid:
                status = "skipped_existing"
                counts["skipped"] += 1
                print(f"[{index}/{len(participants)}] SKIPPED PREPROCESSING: {participant_id} ({reason})")
                print("NOTE: Structural validity does not prove this file was created without quality filtering.")
            else:
                status = "invalid_existing"
                counts["invalid_existing"] += 1
                error = reason
                print(f"[{index}/{len(participants)}] EXISTING FILE NEEDS REVIEW: {participant_id}: {reason}")
                print("File preserved. Rename or remove it manually if you want to reprocess.")
        else:
            print(f"[{index}/{len(participants)}] PROCESSING: {participant_id}", flush=True)
            t0 = time.time()
            try:
                process_one(edf_path, output_dir, metadata_path, args)
                elapsed = (time.time() - t0) / 60
                valid, reason = validate_existing_parquet(output_file, participant_id)
                if not valid:
                    raise RuntimeError(f"Output validation failed: {reason}")
                status = "processed"
                counts["processed"] += 1
                print(f"[{index}/{len(participants)}] COMPLETED: {participant_id} ({elapsed:.2f} min)")
            except KeyboardInterrupt:
                print("Interrupted by user. Saving summaries before exit.")
                save_summaries(output_dir, processing, inspection, gaps)
                raise
            except Exception as exc:
                status = "failed"
                error = str(exc)
                counts["failed"] += 1
                print(f"[{index}/{len(participants)}] FAILED: {participant_id}: {exc}")
                traceback.print_exc()

        processing.append({
            "participant": participant_id,
            "status": status,
            "processing_minutes": round(elapsed, 3),
            "output_file": output_file.name if output_file.exists() else "",
            "error": error,
        })

        if status in ("processed", "skipped_existing"):
            try:
                summary, gap_details = inspect_one(output_file, args.sample_frequency)
                inspection.append(summary)
                gaps.extend(gap_details)
            except Exception as exc:
                counts["inspection_failed"] += 1
                print(f"INSPECTION FAILED for {participant_id}: {exc}")
                traceback.print_exc()
        save_summaries(output_dir, processing, inspection, gaps)

    print("\n" + "=" * 60)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"EDF participants found: {len(participants)}")
    print(f"Newly processed: {counts['processed']}")
    print(f"Skipped existing: {counts['skipped']}")
    print(f"Failed processing: {counts['failed']}")
    print(f"Existing files requiring review: {counts['invalid_existing']}")
    print(f"Inspection failures: {counts['inspection_failed']}")
    print(f"Successfully inspected: {len(inspection)}")
    print(f"Total inspected samples: {sum(x['samples'] for x in inspection):,}")
    print(f"Total timestamp gaps: {len(gaps):,}")
    print(f"Elapsed time: {(time.time() - started) / 60:.2f} minutes")
    print("Participant statuses:")
    for record in processing:
        print(f"  {record['participant']}: {record['status']}")
    print(f"Summaries saved under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EDF participants without quality filtering; resume existing results")
    parser.add_argument("--sample_frequency", type=int, default=1000)
    parser.add_argument("--method", type=str, default="neurokit")
    parser.add_argument("--number_processors", type=int, default=1)
    parser.add_argument("--participant_number", type=int, default=-1,
                        help="-1 selects all; actual participant IDs are extracted from EDF filenames")
    parser.add_argument("--data_chunk", type=int, default=-1,
                        help="-1 searches all raw EDF folders recursively; 1..5 restricts to a part")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.number_processors == -1:
        args.number_processors = os.cpu_count() or 1
    main(args)
