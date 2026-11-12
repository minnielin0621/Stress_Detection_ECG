
# ECG Feature Extraction
# Quality Filter > 0.25
# Process one participant at a time
# Original 32 / Full 55 Features

import os
import glob
import gc
import time
import argparse
import traceback

import pandas as pd
import numpy as np

from collections import Counter

from sia import Segmenter
from sia.io import read_csv, write_csv
from sia.segmenters import SlidingWindow

from sia.features import (
    extract_peaks,
    extract_hr_from_peaks,
    delineate,
    Waves
)

from sia.features.time_domain import (
    hr,
    hrv,
    time_domain,
    Statistic,
    Feature as TimeFeature
)

from sia.features.frequency_domain import (
    frequency_domain
)

from sia.features.nonlinear_domain import (
    nonlinear_domain,
    Feature as NonlinearFeature
)

from sia.features.morphology_domain import (
    morphology_domain,
    Feature as MorphologyFeature
)

from utils.helper_path import (
    CLEANED_DATA_PATH,
    FEATURE_DATA_PATH
)

from utils.helper_functions import create_directory


# ============================================================
# 1. Extract features for one participant
# ============================================================

def extract_one_participant(
    input_file_path,
    output_path,
    args,
    skip_log_dir
):

    participant_id = os.path.splitext(
        os.path.basename(input_file_path)
    )[0]

    WINDOW_SIZE = int(
        args.window_size * args.sample_frequency
    )

    STEP_SIZE = int(
        args.window_shift * args.sample_frequency
    )

    # --------------------------------------------------------
    # Feature extraction pipeline
    # --------------------------------------------------------

    (
        Segmenter()

        # Read only ONE participant
        .data(
            read_csv(
                input_file_path,
                columns=[
                    "timestamp",
                    "ECG_Clean",
                    "ECG_R_Peaks",
                    "category",
                    "label"
                ]
            )
        )

        # Sliding window
        .segment(
            SlidingWindow(
                WINDOW_SIZE,
                STEP_SIZE
            )
        )

        # Save individual skip statistics
        .set_log_file(
            os.path.join(
                skip_log_dir,
                f"{participant_id}_skip_statistics.json"
            )
        )

        # ----------------------------------------------------
        # Skip invalid windows
        # ----------------------------------------------------

        .skip(
            lambda category:
                len(set(category)) > 1,
            "mixed_category_skip"
        )

        .skip(
            lambda label:
                len(set(label)) > 1,
            "mixed_label_skip"
        )

        .skip(
            lambda ECG_R_Peaks:
                len(
                    extract_peaks(ECG_R_Peaks)
                ) < 12,
            "insufficient_peaks_skip"
        )

        .skip(
            lambda ECG_R_Peaks:
                min(
                    extract_hr_from_peaks(
                        ECG_R_Peaks,
                        sample_frequency=args.sample_frequency
                    )
                ) < 40,
            "low_hr_skip"
        )

        .skip(
            lambda ECG_R_Peaks:
                max(
                    extract_hr_from_peaks(
                        ECG_R_Peaks,
                        sample_frequency=args.sample_frequency
                    )
                ) > 220,
            "high_hr_skip"
        )

        # ----------------------------------------------------
        # Extract category and label
        # ----------------------------------------------------

        .extract(
            "category",
            lambda category:
                Counter(category).most_common(1)[0][0]
        )

        .extract(
            "label",
            lambda label:
                Counter(label).most_common(1)[0][0]
        )

        # ----------------------------------------------------
        # Extract R peaks
        # ----------------------------------------------------

        .use(
            "rpeaks",
            lambda ECG_R_Peaks:
                extract_peaks(ECG_R_Peaks)
        )

        # ----------------------------------------------------
        # Heart Rate Features
        # ----------------------------------------------------

        .extract(
            hr(
                [
                    Statistic.MIN,
                    Statistic.MAX,
                    Statistic.MEAN,
                    Statistic.STD
                ],
                sampling_rate=args.sample_frequency
            )
        )

        # ----------------------------------------------------
        # Time Domain Features
        # ----------------------------------------------------

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
                    TimeFeature.NK_CVSD
                ],
                sampling_rate=args.sample_frequency
            )
        )

        # ----------------------------------------------------
        # Frequency Domain Features
        #
        # Original: 1 frequency feature
        # Full: 24 frequency features
        # ----------------------------------------------------

        .extract(
            frequency_domain(
                sampling_rate=args.sample_frequency,
                output_all=args.full_features
            )
        )

        # ----------------------------------------------------
        # Nonlinear Domain Features
        # ----------------------------------------------------

        .extract(
            nonlinear_domain(
                [
                    NonlinearFeature.DFA,
                    NonlinearFeature.ENTROPY,
                    NonlinearFeature.POINCARE,
                    NonlinearFeature.RQA,
                    NonlinearFeature.FRAGMENTATION,
                    NonlinearFeature.HEART_ASYMMETRY
                ],
                sampling_rate=args.sample_frequency
            )
        )

        # ----------------------------------------------------
        # Extract T peaks
        # ----------------------------------------------------

        .use(
            "tpeaks",
            lambda ECG_Clean:
                extract_peaks(
                    delineate(
                        Waves.T_Peak
                    )(
                        ECG_Clean,
                        sampling_rate=args.sample_frequency
                    )
                )
        )

        # ----------------------------------------------------
        # Morphology Domain Features
        # ----------------------------------------------------

        .extract(
            morphology_domain(
                [
                    MorphologyFeature.TWA
                ],
                sampling_rate=args.sample_frequency
            )
        )

        # ----------------------------------------------------
        # Save participant CSV
        # ----------------------------------------------------

        .to(
            write_csv(
                os.path.join(
                    output_path,
                    "[0-9]{5}.csv"
                ),
                use_parquet=False
            )
        )
    )


# ============================================================
# 2. Format processing time
# ============================================================

def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# 3. Save progress summary
# ============================================================

def save_progress(progress_records, output_path):

    progress_df = pd.DataFrame(
        progress_records
    )

    progress_df.to_csv(
        os.path.join(
            output_path,
            "progress_summary.csv"
        ),
        index=False
    )


# ============================================================
# 4. Main
# ============================================================

def main(args):

    print("=" * 65)
    print("ECG FEATURE EXTRACTION")
    print("Quality Filter > 0.25")
    print("Process one participant at a time")
    print("=" * 65)

    feature_count = (
        55 if args.full_features else 32
    )

    print("Sample frequency:", args.sample_frequency)
    print("Window size:", args.window_size)
    print("Window shift:", args.window_shift)
    print("Feature count:", feature_count)

    # --------------------------------------------------------
    # Input path
    #
    # Quality Filter:
    # data/cleaned/1000/
    # --------------------------------------------------------

    input_path = os.path.join(
        CLEANED_DATA_PATH,
        str(args.sample_frequency)
    )

    # --------------------------------------------------------
    # Output path
    #
    # Original 32:
    # data/features/1000/quality_filter/30/10.0full/
    #
    # Full 55:
    # data/features/1000/quality_filter_full_features/30/10.0full/
    # --------------------------------------------------------

    feature_mode = (
        "quality_filter_full_features"
        if args.full_features
        else "quality_filter"
    )

    output_path = os.path.join(
        FEATURE_DATA_PATH,
        str(args.sample_frequency),
        feature_mode,
        str(args.window_size),
        f"{args.window_shift}full"
    )

    create_directory(output_path)

    # Individual skip statistics
    skip_log_dir = os.path.join(
        output_path,
        "skip_logs"
    )

    create_directory(skip_log_dir)

    error_log_path = os.path.join(
        output_path,
        "error_log.txt"
    )

    print("\nInput path:")
    print(input_path)

    print("\nOutput path:")
    print(output_path)

    # --------------------------------------------------------
    # Find participant files
    # Only list file paths, not load all data
    # --------------------------------------------------------

    if args.participant_number == -1:

        input_files = sorted(
            glob.glob(
                os.path.join(
                    input_path,
                    "[0-9][0-9][0-9][0-9][0-9].parquet"
                )
            )
        )

    else:

        participant_file = os.path.join(
            input_path,
            f"{args.participant_number}.parquet"
        )

        if os.path.isfile(participant_file):

            input_files = [participant_file]

        else:

            input_files = []

    if not input_files:

        raise FileNotFoundError(
            f"No participant files found:\n{input_path}"
        )

    total_participants = len(input_files)

    print(
        "\nTotal participants:",
        total_participants
    )

    # --------------------------------------------------------
    # Progress variables
    # --------------------------------------------------------

    completed = 0
    skipped = 0
    failed = 0

    processing_times = []
    progress_records = []

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Process ONE participant at a time
    # --------------------------------------------------------

    for index, input_file_path in enumerate(
        input_files,
        start=1
    ):

        participant_id = os.path.splitext(
            os.path.basename(input_file_path)
        )[0]

        output_csv = os.path.join(
            output_path,
            f"{participant_id}.csv"
        )

        print("\n" + "=" * 65)

        print(
            f"[{index}/{total_participants}] "
            f"Participant {participant_id}"
        )

        print("=" * 65)

        # ----------------------------------------------------
        # Skip existing participant CSV
        # ----------------------------------------------------

        if (
            os.path.isfile(output_csv)
            and not args.overwrite
        ):

            print(
                f"Already exists: {output_csv}"
            )

            print("Skipping participant.")

            skipped += 1

            progress_records.append({
                "participant_id": participant_id,
                "status": "skipped",
                "seconds": 0,
                "error": ""
            })

            save_progress(
                progress_records,
                output_path
            )

            continue

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        participant_start_time = (
            time.perf_counter()
        )

        try:

            print("Reading:")
            print(input_file_path)

            print(
                f"Extracting {feature_count} features..."
            )

            extract_one_participant(
                input_file_path=input_file_path,
                output_path=output_path,
                args=args,
                skip_log_dir=skip_log_dir
            )

            # Check that output was created
            if not os.path.isfile(output_csv):

                raise FileNotFoundError(
                    f"Output CSV not found: {output_csv}"
                )

            elapsed = (
                time.perf_counter()
                - participant_start_time
            )

            processing_times.append(elapsed)

            completed += 1

            print("\nExtraction completed.")

            print(
                "Processing time:",
                format_time(elapsed)
            )

            print("Saved to:")
            print(output_csv)

            progress_records.append({
                "participant_id": participant_id,
                "status": "completed",
                "seconds": round(elapsed, 2),
                "error": ""
            })

        # ----------------------------------------------------
        # Handle Ctrl+C
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print(
                "\nExtraction interrupted by user."
            )

            print(
                "Completed participants remain saved."
            )

            progress_records.append({
                "participant_id": participant_id,
                "status": "interrupted",
                "seconds": round(
                    time.perf_counter()
                    - participant_start_time,
                    2
                ),
                "error": "KeyboardInterrupt"
            })

            save_progress(
                progress_records,
                output_path
            )

            # Remove potentially incomplete CSV
            if os.path.isfile(output_csv):

                os.remove(output_csv)

            print(
                "Run the same command to continue."
            )

            return

        # ----------------------------------------------------
        # Record errors and continue
        # ----------------------------------------------------

        except Exception as e:

            failed += 1

            elapsed = (
                time.perf_counter()
                - participant_start_time
            )

            print("\nERROR:")
            print(str(e))

            print(
                "Moving to the next participant."
            )

            with open(
                error_log_path,
                "a",
                encoding="utf-8"
            ) as f:

                f.write("\n" + "=" * 65 + "\n")

                f.write(
                    f"Participant: {participant_id}\n"
                )

                f.write(
                    f"Input file: {input_file_path}\n"
                )

                f.write(
                    traceback.format_exc()
                )

                f.write("\n")

            # Remove potentially incomplete CSV
            if os.path.isfile(output_csv):

                os.remove(output_csv)

            progress_records.append({
                "participant_id": participant_id,
                "status": "failed",
                "seconds": round(elapsed, 2),
                "error": str(e)
            })

        # ----------------------------------------------------
        # Release memory
        # ----------------------------------------------------

        finally:

            gc.collect()

        # ----------------------------------------------------
        # Save progress
        # ----------------------------------------------------

        save_progress(
            progress_records,
            output_path
        )

        # ----------------------------------------------------
        # Estimate remaining time
        # ----------------------------------------------------

        remaining = (
            total_participants - index
        )

        if processing_times:

            average_time = (
                sum(processing_times)
                / len(processing_times)
            )

            estimated_remaining = (
                average_time * remaining
            )

            print("\nProgress:")

            print(
                f"Processed: "
                f"{index}/{total_participants}"
            )

            print("Completed:", completed)

            print("Skipped:", skipped)

            print("Failed:", failed)

            print(
                "Estimated remaining:",
                format_time(estimated_remaining)
            )

    # --------------------------------------------------------
    # Optional: add demographic data
    # --------------------------------------------------------

    if args.add_static_data:

        add_static_data(
            FEATURE_DATA_PATH,
            output_path,
            args.participant_number
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - total_start_time
    )

    print("\n" + "=" * 65)

    print("FEATURE EXTRACTION FINISHED")

    print("=" * 65)

    print(
        "Total participants:",
        total_participants
    )

    print("Completed:", completed)

    print("Skipped:", skipped)

    print("Failed:", failed)

    print(
        "Total processing time:",
        format_time(total_elapsed)
    )

    print("\nOutput directory:")
    print(output_path)

    print("\nProgress summary:")

    print(
        os.path.join(
            output_path,
            "progress_summary.csv"
        )
    )

    if failed > 0:

        print("\nSome participants failed.")

        print(
            "Check error_log.txt and rerun "
            "to retry failed participants."
        )


# ============================================================
# 5. Optional: Add static data
# ============================================================

def add_static_data(
    FEATURE_DATA_PATH,
    output_path,
    participant_number=30100
):

    demographics_pd = pd.read_csv(
        os.path.join(
            FEATURE_DATA_PATH,
            "demographics_v2.csv"
        )
    )

    if participant_number == -1:

        csv_files = glob.glob(
            os.path.join(
                output_path,
                "[0-9][0-9][0-9][0-9][0-9].csv"
            )
        )

    else:

        csv_files = glob.glob(
            os.path.join(
                output_path,
                f"{participant_number}.csv"
            )
        )

    missing_ids = []

    for file in csv_files:

        participant_number_csv = (
            os.path.basename(file)
        )

        participant_features_pd = (
            pd.read_csv(file)
        )

        participant_number_id = (
            os.path.splitext(
                participant_number_csv
            )[0]
        )

        participant_row = demographics_pd[
            demographics_pd["Subject_ID"]
            == int(participant_number_id)
        ]

        try:

            participant_age = (
                participant_row["Age"].values[0]
            )

            participant_gender = (
                participant_row["Sex"].values[0]
            )

            print(
                "Add static data (gender, age) "
                f"to participant: {participant_number_id}"
            )

        except IndexError:

            missing_ids.append(
                participant_number_id
            )

            participant_age = np.nan

            participant_gender = np.nan

        participant_features_pd["age"] = (
            participant_age
        )

        participant_features_pd["gender"] = (
            participant_gender
        )

        participant_features_pd.to_csv(
            file,
            index=False
        )

    if missing_ids:

        print(
            "Missing demographics:",
            missing_ids
        )


# ============================================================
# 6. Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract ECG features from quality-filtered data "
            "one participant at a time."
        )
    )

    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1000,
        help="Sampling rate used for the dataset."
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Window size in seconds."
    )

    parser.add_argument(
        "--window_shift",
        type=float,
        default=10.0,
        help="Shift in seconds between consecutive windows."
    )

    parser.add_argument(
        "--participant_number",
        type=int,
        default=-1,
        help="Participant ID. Set -1 for all participants."
    )

    parser.add_argument(
        "--full_features",
        action="store_true",
        help="Extract full 55 ECG features."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract existing participant CSV files."
    )

    parser.add_argument(
        "--add_static_data",
        action="store_true",
        help="Add age and gender to participant CSV files."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
