
# Feature extraction without ECG quality threshold filtering

import os
import glob
import argparse
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


def main(args):

    print("=" * 60)
    print("ECG FEATURE EXTRACTION - NO QUALITY FILTER")
    print("=" * 60)

    window_size = int(
        args.window_size * args.sample_frequency
    )

    step_size = int(
        args.window_shift * args.sample_frequency
    )

    # Input directory
    input_path = os.path.join(
        CLEANED_NO_QUALITY_FILTER_PATH,
        str(args.sample_frequency),
    )

    # Output directory
    output_path = os.path.join(
        FEATURE_DATA_PATH,
        str(args.sample_frequency),
        "no_quality_filter",
        str(args.window_size),
        f"{args.window_shift}full",
    )

    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Sampling frequency: {args.sample_frequency} Hz")
    print(f"Window size: {args.window_size} seconds")
    print(f"Window shift: {args.window_shift} seconds")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(
            f"Input directory not found: {input_path}"
        )

    create_directory(output_path)

    if args.participant_number == -1:
        input_file = "*.parquet"
        print("Participant selection: ALL")
    else:
        input_file = f"{args.participant_number}.parquet"
        print(f"Participant selection: {args.participant_number}")

    input_pattern = os.path.join(input_path, input_file)
    matching_files = sorted(glob.glob(input_pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"No matching Parquet files found: {input_pattern}"
        )

    print(f"Matching input files: {len(matching_files)}")
    print("Starting feature extraction...")

    Segmenter() \
        .data(
            read_csv(
                input_pattern,
                columns=[
                    "timestamp",
                    "ECG_Clean",
                    "ECG_R_Peaks",
                    "category",
                    "label",
                ],
            )
        ) \
        .segment(
            SlidingWindow(window_size, step_size)
        ) \
        .set_log_file(
            os.path.join(output_path, "skip_statistics.json")
        ) \
        .skip(
            lambda category: len(set(category)) > 1,
            "mixed_category_skip",
        ) \
        .skip(
            lambda label: len(set(label)) > 1,
            "mixed_label_skip",
        ) \
        .skip(
            lambda ECG_R_Peaks: (
                len(extract_peaks(ECG_R_Peaks)) < 12
            ),
            "insufficient_peaks_skip",
        ) \
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
        ) \
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
        ) \
        .extract(
            "category",
            lambda category: Counter(category).most_common(1)[0][0],
        ) \
        .extract(
            "label",
            lambda label: Counter(label).most_common(1)[0][0],
        ) \
        .use(
            "rpeaks",
            lambda ECG_R_Peaks: extract_peaks(ECG_R_Peaks),
        ) \
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
        ) \
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
        ) \
        .extract(
            frequency_domain(
                sampling_rate=args.sample_frequency
            )
        ) \
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
        ) \
        .use(
            "tpeaks",
            lambda ECG_Clean: extract_peaks(
                delineate(Waves.T_Peak)(
                    ECG_Clean,
                    sampling_rate=args.sample_frequency,
                )
            ),
        ) \
        .extract(
            morphology_domain(
                [MorphologyFeature.TWA],
                sampling_rate=args.sample_frequency,
            )
        ) \
        .to(
            write_csv(
                os.path.join(output_path, "[0-9]{5}.csv"),
                use_parquet=False,
            )
        )

    print("=" * 60)
    print("Feature extraction completed.")
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
        "--add_static_data",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_path = main(args)

    if args.add_static_data:
        add_static_data(
            FEATURE_DATA_PATH,
            output_path,
            args.participant_number,
        )
