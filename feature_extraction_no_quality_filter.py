
# Feature extraction without ECG quality threshold filtering

import os
import glob
import argparse
import time
import traceback
from datetime import timedelta
from collections import Counter

import pandas as pd
import numpy as np

from sia import Segmenter
from sia.io import read_csv, write_csv
from sia.segmenters import SlidingWindow

from sia.features import (
    extract_peaks,
    extract_hr_from_peaks,
    delineate,
    Waves,
)

from sia.features.time_domain import (
    hr,
    time_domain,
    Statistic,
    Feature as TimeFeature,
)

from sia.features.frequency_domain import frequency_domain

from sia.features.nonlinear_domain import (
    nonlinear_domain,
    Feature as NonlinearFeature,
)

from sia.features.morphology_domain import (
    morphology_domain,
    Feature as MorphologyFeature,
)

from utils.helper_path import (
    CLEANED_NO_QUALITY_FILTER_PATH,
    FEATURE_DATA_PATH,
)

from utils.helper_functions import create_directory


def format_duration(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


def extract_one_participant(input_file_path, output_path, args, skip_log_dir):
    """Run the original feature extraction pipeline for exactly one input file."""
    participant_id = os.path.splitext(os.path.basename(input_file_path))[0]
    window_size = int(args.window_size * args.sample_frequency)
    step_size = int(args.window_shift * args.sample_frequency)
    (
        Segmenter()
        .data(
            read_csv(
                input_file_path,
                columns=[
                    "timestamp",
                    "ECG_Clean",
                    "ECG_R_Peaks",
                    "category",
                    "label",
                ],
            )
        )
        .segment(
            SlidingWindow(window_size, step_size)
        )
        .set_log_file(
            os.path.join(skip_log_dir, f"{participant_id}_skip_statistics.json")
        )
        .skip(
            lambda category: len(set(category)) > 1,
            "mixed_category_skip",
        )
        .skip(
            lambda label: len(set(label)) > 1,
            "mixed_label_skip",
        )
        .skip(
            lambda ECG_R_Peaks: (
                len(extract_peaks(ECG_R_Peaks)) < 12
            ),
            "insufficient_peaks_skip",
        )
        .skip(
            lambda ECG_R_Peaks: (
                min(
                    extract_hr_from_peaks(
                        ECG_R_Peaks,
                        sample_frequency=args.sample_frequency,
                    )
                ) < 40
            ),
            "low_hr_skip",
        )
        .skip(
            lambda ECG_R_Peaks: (
                max(
                    extract_hr_from_peaks(
                        ECG_R_Peaks,
                        sample_frequency=args.sample_frequency,
                    )
                ) > 220
            ),
            "high_hr_skip",
        )
        .extract(
            "category",
            lambda category: Counter(category).most_common(1)[0][0],
        )
        .extract(
            "label",
            lambda label: Counter(label).most_common(1)[0][0],
        )
        .use(
            "rpeaks",
            lambda ECG_R_Peaks: extract_peaks(ECG_R_Peaks),
        )
        .extract(
            hr(
                [
                    Statistic.MIN,
                    Statistic.MAX,
                    Statistic.MEAN,
                    Statistic.STD,
                ],
                sampling_rate=args.sample_frequency,
            )
        )
        .extract(
            time_domain(
                [
                    TimeFeature.NK_RMSSD,
                    TimeFeature.NK_MeanNN,
                    TimeFeature.NK_SDNN,
                    TimeFeature.NK_MAD_NN,
                    TimeFeature.NK_SD_RMSSD,
                    TimeFeature.NK_IQR_NN,
                    TimeFeature.NN20,
                    TimeFeature.NK_PNN20,
                    TimeFeature.NN50,
                    TimeFeature.NK_PNN50,
                    TimeFeature.NK_CVNN,
                    TimeFeature.NK_CVSD,
                ],
                sampling_rate=args.sample_frequency,
            )
        )
        .extract(
            frequency_domain(
                sampling_rate=args.sample_frequency,
                output_all=args.full_features,
            )
        )
        .extract(
            nonlinear_domain(
                [
                    NonlinearFeature.DFA,
                    NonlinearFeature.ENTROPY,
                    NonlinearFeature.POINCARE,
                    NonlinearFeature.RQA,
                    NonlinearFeature.FRAGMENTATION,
                    NonlinearFeature.HEART_ASYMMETRY,
                ],
                sampling_rate=args.sample_frequency,
            )
        )
        .use(
            "tpeaks",
            lambda ECG_Clean: extract_peaks(
                delineate(Waves.T_Peak)(
                    ECG_Clean,
                    sampling_rate=args.sample_frequency,
                )
            ),
        )
        .extract(
            morphology_domain(
                [MorphologyFeature.TWA],
                sampling_rate=args.sample_frequency,
            )
        )
        .to(
            write_csv(
                os.path.join(output_path, "[0-9]{5}.csv"),
                use_parquet=False,
            )
        )
    )



def main(args):
    print("=" * 60)
    print("ECG FEATURE EXTRACTION - NO QUALITY FILTER")
    print("=" * 60)

    input_path = os.path.join(
        CLEANED_NO_QUALITY_FILTER_PATH,
        str(args.sample_frequency),
    )
    feature_mode = (
        "no_quality_filter_full_features"
        if args.full_features else "no_quality_filter"
    )
    output_path = os.path.join(
        FEATURE_DATA_PATH,
        str(args.sample_frequency),
        feature_mode,
        str(args.window_size),
        f"{args.window_shift}full",
    )

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    input_file = (
        "*.parquet" if args.participant_number == -1
        else f"{args.participant_number}.parquet"
    )
    matching_files = sorted(glob.glob(os.path.join(input_path, input_file)))
    if not matching_files:
        raise FileNotFoundError("No matching Parquet files found")

    create_directory(output_path)
    skip_log_dir = os.path.join(output_path, "skip_logs")
    create_directory(skip_log_dir)
    summary_path = os.path.join(output_path, "progress_summary.csv")

    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Sampling frequency: {args.sample_frequency} Hz")
    print(f"Window size: {args.window_size} seconds")
    print(f"Window shift: {args.window_shift} seconds")
    print(f"Full frequency features: {args.full_features}")
    print(f"Participants: {len(matching_files)}")
    print(f"Existing CSV: {'OVERWRITE' if args.overwrite else 'SKIP'}")

    start = time.perf_counter()
    results = []
    counts = Counter()
    completed = 0

    try:
        for index, input_file_path in enumerate(matching_files, 1):
            participant_id = os.path.splitext(
                os.path.basename(input_file_path)
            )[0]
            csv_path = os.path.join(output_path, f"{participant_id}.csv")
            print(
                f"\n[{index}/{len(matching_files)}] Participant {participant_id} "
                f"| elapsed {format_duration(time.perf_counter() - start)}",
                flush=True,
            )
            subject_start = time.perf_counter()
            status = "FAILED"
            rows = 0
            columns = 0
            error = ""
            try:
                if os.path.isfile(csv_path) and not args.overwrite:
                    # Existing files are considered completed, not recomputed.
                    status = "SKIPPED_EXISTING"
                    print("  SKIP: CSV already exists", flush=True)
                else:
                    print("  Reading and extracting...", flush=True)
                    extract_one_participant(
                        input_file_path, output_path, args, skip_log_dir
                    )
                    if os.path.isfile(csv_path):
                        df = pd.read_csv(csv_path)
                        rows, columns = df.shape
                        if rows:
                            status = "SUCCESS"
                            print(
                                f"  SUCCESS: {rows} rows, {columns} columns",
                                flush=True,
                            )
                        else:
                            status = "NO_VALID_WINDOWS"
                            print("  SKIP: no valid windows", flush=True)
                    else:
                        status = "NO_OUTPUT"
                        print("  SKIP: no CSV generated", flush=True)
            except KeyboardInterrupt:
                print("\nInterrupted during current participant.", flush=True)
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {error}", flush=True)
                traceback.print_exc()
            finally:
                duration = time.perf_counter() - subject_start
                # Interrupted participants are not recorded as completed.
                if not (status == "FAILED" and not error):
                    counts[status] += 1
                    completed += 1
                    results.append({
                        "participant": participant_id,
                        "status": status,
                        "rows": rows,
                        "columns": columns,
                        "seconds": round(duration, 2),
                        "error": error,
                    })
                    pd.DataFrame(results).to_csv(summary_path, index=False)
                    elapsed = time.perf_counter() - start
                    remaining = (
                        elapsed / index * (len(matching_files) - index)
                    )
                    print(
                        f"  Subject time: {format_duration(duration)} | "
                        f"Total: {format_duration(elapsed)} | "
                        f"ETA: {format_duration(remaining)}",
                        flush=True,
                    )
                    print(
                        f"  Counts: success={counts['SUCCESS']}, "
                        f"existing={counts['SKIPPED_EXISTING']}, "
                        f"no-output={counts['NO_VALID_WINDOWS'] + counts['NO_OUTPUT']}, "
                        f"failed={counts['FAILED']}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopped. Previously completed CSV files are retained.")
        print("Run again without --overwrite to skip existing CSVs.")
        raise

    print("=" * 60)
    print(f"Finished {completed}/{len(matching_files)} participants")
    print(f"Total time: {format_duration(time.perf_counter() - start)}")
    print(f"Progress summary: {summary_path}")
    print(f"Output directory: {output_path}")
    print("=" * 60)
    return output_path


def add_static_data(
    feature_root,
    output_path,
    participant_number=-1,
):
    demographics_file = os.path.join(
        feature_root,
        "demographics_v2.csv",
    )

    if not os.path.isfile(demographics_file):
        raise FileNotFoundError(
            f"Demographics file not found: {demographics_file}"
        )

    demographics_pd = pd.read_csv(demographics_file)

    if participant_number == -1:
        csv_files = glob.glob(
            os.path.join(
                output_path,
                "[0-9][0-9][0-9][0-9][0-9].csv",
            )
        )
    else:
        csv_files = glob.glob(
            os.path.join(
                output_path,
                f"{participant_number}.csv",
            )
        )

    missing_ids = []

    for file in csv_files:
        participant_id = os.path.splitext(
            os.path.basename(file)
        )[0]

        participant_features_pd = pd.read_csv(file)

        participant_row = demographics_pd[
            demographics_pd["Subject_ID"] == int(participant_id)
        ]

        if participant_row.empty:
            print(f"Missing demographics: {participant_id}")
            missing_ids.append(participant_id)
            participant_age = np.nan
            participant_gender = np.nan
        else:
            participant_age = participant_row["Age"].iloc[0]
            participant_gender = participant_row["Sex"].iloc[0]

        participant_features_pd["age"] = participant_age
        participant_features_pd["gender"] = participant_gender

        participant_features_pd.to_csv(
            file,
            index=False,
        )

        print(f"Static data added: {participant_id}")

    print(
        f"Participants missing demographics: {len(missing_ids)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract ECG features without quality threshold filtering."
        )
    )

    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--window_shift",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--participant_number",
        type=int,
        default=-1,
        help="Participant ID or -1 for all participants.",
    )

    parser.add_argument(
        "--full_features",
        action="store_true",
        help="Export all 24 HF/VHF/UHF frequency features.",
    )

    parser.add_argument(
        "--add_static_data",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate existing CSVs instead of skipping them.",
    )

    args = parser.parse_args()

    output_path = main(args)

    if args.add_static_data:
        add_static_data(
            FEATURE_DATA_PATH,
            output_path,
            args.participant_number,
        )
