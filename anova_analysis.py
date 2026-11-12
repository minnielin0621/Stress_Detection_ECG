
import os
import glob
import argparse
import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer


# ============================================================
# 1. Configuration
# ============================================================

EXCLUDE_COLUMNS = [
    "category",
    "label",
    "stress_label",
    "participant_id",
    "subject_id",
    "Subject_ID",
    "timestamp",
    "index",
    "age",
    "gender",
    "window_id",
    "start",
    "end",
    "start_time",
    "end_time",
    "Unnamed: 0"
]

CATEGORY_BASELINE = "baseline"
CATEGORY_STRESS = "mental_stress"

PHYSICAL_CATEGORIES = [
    "low_physical_activity",
    "moderate_physical_activity",
    "high_physical_activity"
]

ALL_CATEGORIES = [
    CATEGORY_BASELINE,
    CATEGORY_STRESS
] + PHYSICAL_CATEGORIES


# ============================================================
# 2. Find participant CSV files
# ============================================================

def find_csv_files(directory):

    files = sorted(
        glob.glob(
            os.path.join(
                directory,
                "[0-9][0-9][0-9][0-9][0-9].csv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No participant CSV files found:\n"
            f"{os.path.abspath(directory)}"
        )

    return files


# ============================================================
# 3. Load participant data
# ============================================================

def load_participant_data(directory):

    csv_files = find_csv_files(directory)

    print("\nLoading data from:")
    print(os.path.abspath(directory))

    print("Participants:", len(csv_files))

    data_list = []

    for file in csv_files:

        df = pd.read_csv(file)

        participant_id = os.path.splitext(
            os.path.basename(file)
        )[0]

        df["participant_id"] = participant_id

        data_list.append(df)

        print(
            f"Loaded {participant_id}: "
            f"{len(df)} windows"
        )

    data = pd.concat(
        data_list,
        ignore_index=True
    )

    print("\nTotal windows:", len(data))

    return data


# ============================================================
# 4. Identify ECG feature columns
# ============================================================

def get_feature_columns(data):

    features = data.drop(
        columns=EXCLUDE_COLUMNS,
        errors="ignore"
    )

    features = features.select_dtypes(
        include=[np.number]
    )

    feature_columns = [
        col for col in features.columns
        if not col.startswith("Unnamed:")
    ]

    return feature_columns


# ============================================================
# 5. Get Original 32 feature names
# ============================================================

def get_features_32(directory):

    files = find_csv_files(directory)

    # Read original CSV files to identify numeric features
    feature_names = set()
    ordered_features = []

    for file in files:

        df = pd.read_csv(file)

        columns = get_feature_columns(df)

        for col in columns:

            if col not in feature_names:

                feature_names.add(col)
                ordered_features.append(col)

    return ordered_features


# ============================================================
# 6. Create binary stress labels
# ============================================================

def create_stress_labels(
    data,
    mode,
    exclude_recovery=False
):

    data = data.copy()

    required_columns = [
        "category",
        "label"
    ]

    for col in required_columns:

        if col not in data.columns:

            raise ValueError(
                f"Required column not found: {col}"
            )

    # Normalize category strings
    data["category"] = (
        data["category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Optional removal of recovery windows
    if exclude_recovery:

        recovery_mask = (
            data["label"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.startswith("recov", na=False)
        )

        removed = int(recovery_mask.sum())

        data = data.loc[
            ~recovery_mask
        ].copy()

        print(
            "\nRecovery windows removed:",
            removed
        )

    # --------------------------------------------------------
    # Mode A: mental stress vs baseline
    # --------------------------------------------------------

    if mode == "baseline":

        selected_categories = [
            CATEGORY_BASELINE,
            CATEGORY_STRESS
        ]

    # --------------------------------------------------------
    # Mode B: mental stress vs all other categories
    # --------------------------------------------------------

    elif mode == "all":

        selected_categories = ALL_CATEGORIES

    else:

        raise ValueError(
            f"Unknown analysis mode: {mode}"
        )

    before = len(data)

    data = data[
        data["category"].isin(
            selected_categories
        )
    ].copy()

    print(
        "Excluded windows:",
        before - len(data)
    )

    # mental_stress = 1
    # selected non-mental-stress categories = 0

    data["stress_label"] = (
        data["category"] == CATEGORY_STRESS
    ).astype(int)

    data = data.reset_index(drop=True)

    if data["stress_label"].nunique() != 2:

        raise ValueError(
            "Expected both stress and non-stress classes."
        )

    print("\nAnalysis mode:", mode)

    print("\nCategory distribution:")

    print(
        data["category"]
        .value_counts()
        .to_string()
    )

    print("\nBinary class distribution:")

    print(
        data["stress_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return data


# ============================================================
# 7. Subject-independent split
# ============================================================

def create_subject_split(
    data,
    test_size,
    random_state
):

    groups = data["participant_id"]

    y = data["stress_label"]

    if groups.nunique() < 2:

        raise ValueError(
            "At least two participants are required."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        splitter.split(
            data,
            y,
            groups
        )
    )

    train_subjects = set(
        groups.iloc[train_idx]
    )

    test_subjects = set(
        groups.iloc[test_idx]
    )

    if train_subjects.intersection(test_subjects):

        raise ValueError(
            "Participant leakage detected."
        )

    split_table = pd.DataFrame({
        "participant_id": sorted(
            train_subjects | test_subjects
        )
    })

    split_table["split"] = (
        split_table["participant_id"]
        .map(
            lambda participant:
            "train"
            if participant in train_subjects
            else "test"
        )
    )

    print("\nTraining participants:",
          len(train_subjects))

    print("Testing participants:",
          len(test_subjects))

    return split_table


# ============================================================
# 8. Run ANOVA
# ============================================================

def run_anova(
    X_train,
    y_train,
    feature_columns,
    features_32,
    output_file
):

    print("\n" + "=" * 60)
    print("Running ANOVA")
    print("Input features:", len(feature_columns))
    print("=" * 60)

    X = X_train[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Missing rates before imputation
    missing_rates = X.isna().mean()

    # Observed class means before imputation
    mean_label_0 = X.loc[
        y_train == 0
    ].mean()

    mean_label_1 = X.loc[
        y_train == 1
    ].mean()

    # Remove constant and entirely missing features
    valid_columns = [
        col for col in feature_columns
        if X[col].nunique(dropna=True) > 1
    ]

    invalid_columns = [
        col for col in feature_columns
        if col not in valid_columns
    ]

    print("Valid features:", len(valid_columns))
    print("Invalid features:", len(invalid_columns))

    if invalid_columns:

        print("\nInvalid features:")

        for col in invalid_columns:
            print("  ", col)

    # Create result table for all features
    results = pd.DataFrame({
        "feature": feature_columns
    })

    results["feature_set"] = (
        results["feature"].map(
            lambda col:
            "original"
            if col in features_32
            else "added"
        )
    )

    results["F_score"] = np.nan
    results["p_value"] = np.nan

    # --------------------------------------------------------
    # Imputation and ANOVA on training data only
    # --------------------------------------------------------

    if valid_columns:

        imputer = SimpleImputer(
            strategy="median"
        )

        X_imputed = imputer.fit_transform(
            X[valid_columns]
        )

        f_scores, p_values = f_classif(
            X_imputed,
            y_train
        )

        score_table = pd.DataFrame({
            "feature": valid_columns,
            "F_score": f_scores,
            "p_value": p_values
        }).set_index("feature")

        results["F_score"] = (
            results["feature"]
            .map(score_table["F_score"])
        )

        results["p_value"] = (
            results["feature"]
            .map(score_table["p_value"])
        )

    # --------------------------------------------------------
    # Add statistics
    # --------------------------------------------------------

    results["missing_rate"] = (
        results["feature"]
        .map(missing_rates)
    )

    results["mean_label_0"] = (
        results["feature"]
        .map(mean_label_0)
    )

    results["mean_label_1"] = (
        results["feature"]
        .map(mean_label_1)
    )

    results["mean_difference"] = (
        results["mean_label_1"]
        - results["mean_label_0"]
    )

    results["usable"] = (
        results["F_score"].notna()
        & results["p_value"].notna()
    )

    results["p_below_0.05"] = (
        results["usable"]
        & (results["p_value"] < 0.05)
    )

    # --------------------------------------------------------
    # Rank features
    # --------------------------------------------------------

    results = results.sort_values(
        ["usable", "F_score"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)

    results["rank"] = np.nan

    usable_mask = results["usable"]

    results.loc[
        usable_mask,
        "rank"
    ] = np.arange(
        1,
        usable_mask.sum() + 1
    )

    results = results[
        [
            "rank",
            "feature",
            "feature_set",
            "F_score",
            "p_value",
            "p_below_0.05",
            "mean_label_0",
            "mean_label_1",
            "mean_difference",
            "missing_rate",
            "usable"
        ]
    ]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop 10 features:")

    print(
        results.head(10)
        .to_string(index=False)
    )

    print("\nResults saved to:")
    print(output_file)

    return results


# ============================================================
# 9. Run one analysis mode
# ============================================================

def run_analysis(
    data,
    features_32,
    features_55,
    split_table,
    mode,
    args
):

    print("\n" + "#" * 60)
    print("ANALYSIS MODE:", mode)
    print("#" * 60)

    # Create binary labels
    analysis_data = create_stress_labels(
        data=data,
        mode=mode,
        exclude_recovery=args.exclude_recovery
    )

    # Use the same participant split for both modes
    split_mapping = (
        split_table
        .set_index("participant_id")["split"]
    )

    analysis_data["split"] = (
        analysis_data["participant_id"]
        .map(split_mapping)
    )

    if analysis_data["split"].isna().any():

        raise ValueError(
            "Some participants have no assigned split."
        )

    train_data = analysis_data[
        analysis_data["split"] == "train"
    ].copy()

    test_data = analysis_data[
        analysis_data["split"] == "test"
    ].copy()

    X_train = train_data[features_55]

    y_train = train_data["stress_label"]

    y_test = test_data["stress_label"]

    print("\nTraining windows:", len(train_data))
    print("Testing windows:", len(test_data))

    print("\nTraining class distribution:")

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting class distribution:")

    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    if y_train.nunique() != 2:

        raise ValueError(
            f"Training set for mode '{mode}' "
            "must contain both classes."
        )

    if y_test.nunique() != 2:

        print(
            "\nWARNING: Test set does not contain "
            "both classes."
        )

    # --------------------------------------------------------
    # Create mode-specific output directory
    # --------------------------------------------------------

    output_dir = os.path.join(
        args.output_dir,
        mode
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # ANOVA for Original 32 features
    # --------------------------------------------------------

    results_32 = run_anova(
        X_train=X_train,
        y_train=y_train,
        feature_columns=features_32,
        features_32=features_32,
        output_file=os.path.join(
            output_dir,
            "anova_32.csv"
        )
    )

    # --------------------------------------------------------
    # ANOVA for Full 55 features
    # --------------------------------------------------------

    results_55 = run_anova(
        X_train=X_train,
        y_train=y_train,
        feature_columns=features_55,
        features_32=features_32,
        output_file=os.path.join(
            output_dir,
            "anova_55.csv"
        )
    )

    # --------------------------------------------------------
    # Added 23 features
    # --------------------------------------------------------

    added_results = results_55[
        results_55["feature_set"] == "added"
    ].copy()

    added_results = added_results.sort_values(
        ["usable", "F_score"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)

    added_results["rank_among_added"] = np.nan

    usable_mask = added_results["usable"]

    added_results.loc[
        usable_mask,
        "rank_among_added"
    ] = np.arange(
        1,
        usable_mask.sum() + 1
    )

    added_results.to_csv(
        os.path.join(
            output_dir,
            "anova_added_23.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save class distribution
    # --------------------------------------------------------

    class_distribution = pd.DataFrame({
        "split": [
            "train",
            "train",
            "test",
            "test"
        ],
        "stress_label": [0, 1, 0, 1],
        "windows": [
            int((y_train == 0).sum()),
            int((y_train == 1).sum()),
            int((y_test == 0).sum()),
            int((y_test == 1).sum())
        ]
    })

    class_distribution.to_csv(
        os.path.join(
            output_dir,
            "class_distribution.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save category distribution by split
    # --------------------------------------------------------

    category_distribution = (
        analysis_data.groupby(
            [
                "split",
                "category",
                "stress_label"
            ]
        )
        .size()
        .reset_index(name="windows")
    )

    category_distribution.to_csv(
        os.path.join(
            output_dir,
            "category_distribution.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_mapping = (
        analysis_data[
            [
                "category",
                "label",
                "stress_label"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "stress_label",
                "category",
                "label"
            ]
        )
    )

    label_mapping.to_csv(
        os.path.join(
            output_dir,
            "label_mapping.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Analysis summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ANOVA MODE COMPLETED:", mode)
    print("=" * 60)

    print(
        "Original features:",
        len(features_32)
    )

    print(
        "Full features:",
        len(features_55)
    )

    print(
        "Added features:",
        len(added_results)
    )

    print("\nFeatures with p < 0.05:")

    print(
        "Original 32:",
        results_32["p_below_0.05"].sum()
    )

    print(
        "Full 55:",
        results_55["p_below_0.05"].sum()
    )

    print(
        "Added 23:",
        added_results["p_below_0.05"].sum()
    )

    print("\nTop 10 added features:")

    print(
        added_results.head(10)
        .to_string(index=False)
    )

    print("\nOutput directory:")
    print(os.path.abspath(output_dir))


# ============================================================
# 10. Main function
# ============================================================

def main(args):

    print("=" * 60)
    print("ECG ANOVA Feature Analysis")
    print("Original 32 vs Full 55")
    print("No Random Forest Training")
    print("=" * 60)

    # --------------------------------------------------------
    # Load Full 55 data
    # --------------------------------------------------------

    data = load_participant_data(
        args.features_55_dir
    )

    # Check required columns
    for col in ["category", "label"]:

        if col not in data.columns:

            raise ValueError(
                f"Required column not found: {col}"
            )

    # --------------------------------------------------------
    # Print original categories and labels
    # --------------------------------------------------------

    print("\nOriginal category distribution:")

    print(
        data["category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOriginal task label distribution:")

    print(
        data["label"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Identify 32 and 55 features
    # --------------------------------------------------------

    features_55 = get_feature_columns(
        data
    )

    features_32 = get_features_32(
        args.features_32_dir
    )

    print("\nOriginal feature count:",
          len(features_32))

    print("Full feature count:",
          len(features_55))

    if len(features_32) != 32:

        raise ValueError(
            f"Expected 32 features, "
            f"found {len(features_32)}.\n"
            f"Features: {features_32}"
        )

    if len(features_55) != 55:

        raise ValueError(
            f"Expected 55 features, "
            f"found {len(features_55)}.\n"
            f"Features: {features_55}"
        )

    missing_original = [
        col for col in features_32
        if col not in features_55
    ]

    if missing_original:

        raise ValueError(
            "Original features missing "
            "from Full 55 data:\n"
            f"{missing_original}"
        )

    added_features = [
        col for col in features_55
        if col not in features_32
    ]

    print("\nAdded feature count:",
          len(added_features))

    print("\nAdded feature names:")

    for col in added_features:
        print("  ", col)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save feature inventory
    # --------------------------------------------------------

    feature_inventory = pd.DataFrame({
        "feature": features_55,
        "feature_set": [
            "original"
            if col in features_32
            else "added"
            for col in features_55
        ]
    })

    feature_inventory.to_csv(
        os.path.join(
            args.output_dir,
            "feature_inventory.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Create one shared subject split
    # --------------------------------------------------------

    split_data = create_stress_labels(
        data=data,
        mode="all",
        exclude_recovery=args.exclude_recovery
    )

    split_table = create_subject_split(
        data=split_data,
        test_size=args.test_size,
        random_state=args.random_state
    )

    split_table.to_csv(
        os.path.join(
            args.output_dir,
            "subject_split.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Run selected analysis modes
    # --------------------------------------------------------

    if args.mode == "both":

        modes = [
            "baseline",
            "all"
        ]

    else:

        modes = [args.mode]

    for mode in modes:

        run_analysis(
            data=data,
            features_32=features_32,
            features_55=features_55,
            split_table=split_table,
            mode=mode,
            args=args
        )

    print("\n" + "=" * 60)
    print("ALL ANOVA ANALYSES COMPLETED")
    print("=" * 60)

    print("\nOutput directory:")
    print(os.path.abspath(args.output_dir))

    print(
        "\nNOTE: ANOVA p-values are exploratory "
        "because windows from the same participant "
        "may be correlated and overlapping."
    )

    print(
        "\nNo Random Forest model was trained."
    )


# ============================================================
# 11. Command-line arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "ANOVA analysis for Original 32 "
            "and Full 55 ECG features"
        )
    )

    parser.add_argument(
        "--features_32_dir",
        type=str,
        default=(
            "data/features/1000/"
            "no_quality_filter/"
            "30/10.0full"
        )
    )

    parser.add_argument(
        "--features_55_dir",
        type=str,
        default=(
            "data/features/1000/"
            "no_quality_filter_full_features/"
            "30/10.0full"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/results/anova_30s"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "baseline",
            "all",
            "both"
        ],
        default="both"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--exclude_recovery",
        action="store_true",
        help="Exclude tasks whose label starts with Recov"
    )

    args = parser.parse_args()

    main(args)
