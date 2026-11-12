
import os
import glob
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. Exclude non-feature columns
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
    "source_row",
    "Unnamed: 0"
]


# ============================================================
# 2. Load participant CSV files
# ============================================================

def load_data(input_dir):

    files = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "[0-9][0-9][0-9][0-9][0-9].csv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found:\n{input_dir}"
        )

    print("\nLoading data:")
    print(input_dir)

    data_list = []

    for file in files:

        df = pd.read_csv(file)

        participant_id = os.path.splitext(
            os.path.basename(file)
        )[0]

        df["participant_id"] = participant_id

        df["source_row"] = np.arange(len(df))

        data_list.append(df)

    data = pd.concat(
        data_list,
        ignore_index=True
    )

    print("Participants:",
          data["participant_id"].nunique())

    print("Total windows:", len(data))

    return data


# ============================================================
# 3. Get ECG feature columns
# ============================================================

def get_feature_columns(data):

    features = data.drop(
        columns=EXCLUDE_COLUMNS,
        errors="ignore"
    )

    features = features.select_dtypes(
        include=[np.number]
    )

    features = features.loc[
        :,
        ~features.columns.str.startswith("Unnamed:")
    ]

    return features.columns.tolist()


# ============================================================
# 4. Create binary labels
# ============================================================

def create_stress_labels(data):

    data = data.copy()

    data["category"] = (
        data["category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Only keep mental stress and baseline
    valid_categories = [
        "mental_stress",
        "baseline"
    ]

    before = len(data)

    data = data.loc[
        data["category"].isin(valid_categories)
    ].copy()

    print("\nExcluded physical activity windows:",
          before - len(data))

    # mental_stress = 1
    # baseline = 0

    data["stress_label"] = (
        data["category"] == "mental_stress"
    ).astype(int)

    data = data.reset_index(drop=True)

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
# 5. Subject-independent train/test split
# ============================================================

def split_data(data, args):

    y = data["stress_label"]

    groups = data["participant_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state
    )

    train_idx, test_idx = next(
        splitter.split(data, y, groups)
    )

    train_data = data.iloc[
        train_idx
    ].copy()

    test_data = data.iloc[
        test_idx
    ].copy()

    train_subjects = set(
        train_data["participant_id"]
    )

    test_subjects = set(
        test_data["participant_id"]
    )

    # Check subject leakage
    overlap = train_subjects.intersection(
        test_subjects
    )

    if overlap:
        raise ValueError(
            f"Subject leakage detected: {overlap}"
        )

    if train_data["stress_label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    if test_data["stress_label"].nunique() != 2:
        raise ValueError(
            "Testing data must contain both classes."
        )

    print("\nTraining participants:",
          len(train_subjects))

    print("Testing participants:",
          len(test_subjects))

    print("Training windows:", len(train_data))

    print("Testing windows:", len(test_data))

    # Save subject split
    split_table = pd.DataFrame({

        "participant_id":
            sorted(train_subjects)
            + sorted(test_subjects),

        "split":
            ["train"] * len(train_subjects)
            + ["test"] * len(test_subjects)
    })

    split_table.to_csv(
        os.path.join(
            args.output_dir,
            "subject_split.csv"
        ),
        index=False
    )

    return train_data, test_data


# ============================================================
# 6. Train one SVM model
# ============================================================

def train_svm(
    train_data,
    test_data,
    features,
    feature_count,
    args
):

    print("\n" + "=" * 60)

    print(f"Training SVM with {feature_count} features")

    print("=" * 60)

    output_dir = os.path.join(
        args.output_dir,
        f"svm_{feature_count}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Prepare features and labels
    # --------------------------------------------------------

    X_train = train_data[
        features
    ].copy()

    X_test = test_data[
        features
    ].copy()

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y_train = train_data[
        "stress_label"
    ].astype(int)

    y_test = test_data[
        "stress_label"
    ].astype(int)

    # Reject features entirely missing in training
    all_missing = X_train.columns[
        X_train.isna().all()
    ].tolist()

    if all_missing:
        raise ValueError(
            f"All-missing training features: {all_missing}"
        )

    print("Training shape:", X_train.shape)

    print("Testing shape:", X_test.shape)

    print("Training missing values:",
          X_train.isna().sum().sum())

    print("Testing missing values:",
          X_test.isna().sum().sum())

    # --------------------------------------------------------
    # Build SVM pipeline
    # --------------------------------------------------------

    model = Pipeline([

        # Fill missing values using training median
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        # Standardize ECG features
        (
            "scaler",
            StandardScaler()
        ),

        # RBF Support Vector Machine
        (
            "svm",
            SVC(
                kernel="rbf",
                C=args.C,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                cache_size=args.cache_size,
                verbose=True,
                random_state=args.random_state
            )
        )
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining SVM...")

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    print("\nTraining completed.")

    print(
        f"Training time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Predict unseen test participants
    # --------------------------------------------------------

    print("\nPredicting test data...")

    y_pred = model.predict(
        X_test
    )

    # SVM decision scores for ROC-AUC
    y_score = model.decision_function(
        X_test
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n===== SVM Results =====")

    print("Features:", feature_count)

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1-score: {f1:.4f}")

    print(f"ROC-AUC: {auc:.4f}")

    print(f"False Positive Rate: {fpr:.4f}")

    print("\nConfusion Matrix:")

    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-stress",
                "Stress"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {

        "model": "SVM",

        "kernel": "rbf",

        "features": feature_count,

        "C": args.C,

        "accuracy": accuracy,

        "balanced_accuracy": balanced_acc,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": auc,

        "false_positive_rate": fpr,

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),

        "training_windows": len(X_train),

        "testing_windows": len(X_test),

        "training_seconds": training_seconds
    }

    pd.DataFrame(
        [results]
    ).to_csv(

        os.path.join(
            output_dir,
            "results.csv"
        ),

        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    pd.DataFrame(

        cm,

        index=[
            "Actual_NonStress",
            "Actual_Stress"
        ],

        columns=[
            "Predicted_NonStress",
            "Predicted_Stress"
        ]

    ).to_csv(

        os.path.join(
            output_dir,
            "confusion_matrix.csv"
        )
    )

    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    predictions = test_data[[
        "participant_id",
        "source_row",
        "category",
        "label",
        "stress_label"
    ]].copy()

    predictions = predictions.rename(
        columns={
            "stress_label": "y_true"
        }
    )

    predictions["y_pred"] = y_pred

    predictions["decision_score"] = y_score

    conditions = [

        (
            (predictions["y_true"] == 0)
            & (predictions["y_pred"] == 0)
        ),

        (
            (predictions["y_true"] == 0)
            & (predictions["y_pred"] == 1)
        ),

        (
            (predictions["y_true"] == 1)
            & (predictions["y_pred"] == 0)
        ),

        (
            (predictions["y_true"] == 1)
            & (predictions["y_pred"] == 1)
        )
    ]

    predictions["prediction_type"] = np.select(
        conditions,
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN"
    )

    predictions.to_csv(

        os.path.join(
            output_dir,
            "predictions.csv"
        ),

        index=False
    )

    # Save false positives separately
    false_positives = predictions.loc[
        predictions["prediction_type"] == "FP"
    ]

    false_positives.to_csv(

        os.path.join(
            output_dir,
            "false_positives.csv"
        ),

        index=False
    )

    # Save feature names
    pd.DataFrame({

        "feature": features

    }).to_csv(

        os.path.join(
            output_dir,
            "features.csv"
        ),

        index=False
    )

    print("\nResults saved to:")

    print(os.path.abspath(output_dir))

    return results


# ============================================================
# 7. Main
# ============================================================

def main(args):

    print("=" * 60)

    print("ECG STRESS CLASSIFICATION - SVM")

    print("Original 32 vs Full 55 Features")

    print("Stress vs Baseline")

    print("=" * 60)

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load original 32-feature data
    # --------------------------------------------------------

    data_32 = load_data(
        args.features_32_dir
    )

    features_32 = get_feature_columns(
        data_32
    )

    print("\nOriginal feature count:",
          len(features_32))

    if len(features_32) != 32:
        raise ValueError(
            f"Expected 32 features, "
            f"found {len(features_32)}"
        )

    # Original data is only used to identify feature names
    del data_32

    # --------------------------------------------------------
    # Load full 55-feature data
    # --------------------------------------------------------

    data_55 = load_data(
        args.features_55_dir
    )

    features_55 = get_feature_columns(
        data_55
    )

    print("\nFull feature count:",
          len(features_55))

    if len(features_55) != 55:
        raise ValueError(
            f"Expected 55 features, "
            f"found {len(features_55)}"
        )

    # Check that original 32 features exist in full 55
    missing_features = set(
        features_32
    ) - set(
        features_55
    )

    if missing_features:
        raise ValueError(
            f"Original features missing in full data: "
            f"{missing_features}"
        )

    print("\nAdded features:",
          len(set(features_55) - set(features_32)))

    # --------------------------------------------------------
    # Use the same full dataset for both experiments
    # --------------------------------------------------------

    data = create_stress_labels(
        data_55
    )

    # Split once and reuse for both models
    train_data, test_data = split_data(
        data,
        args
    )

    # --------------------------------------------------------
    # Train SVM 32
    # --------------------------------------------------------

    print("\n[1/2] Running SVM 32...")

    result_32 = train_svm(

        train_data=train_data,

        test_data=test_data,

        features=features_32,

        feature_count=32,

        args=args
    )

    # Save partial result immediately
    pd.DataFrame(
        [result_32]
    ).to_csv(

        os.path.join(
            args.output_dir,
            "comparison.csv"
        ),

        index=False
    )

    # --------------------------------------------------------
    # Train SVM 55
    # --------------------------------------------------------

    print("\n[2/2] Running SVM 55...")

    result_55 = train_svm(

        train_data=train_data,

        test_data=test_data,

        features=features_55,

        feature_count=55,

        args=args
    )

    # --------------------------------------------------------
    # Compare both models
    # --------------------------------------------------------

    comparison = pd.DataFrame([
        result_32,
        result_55
    ])

    comparison.to_csv(

        os.path.join(
            args.output_dir,
            "comparison.csv"
        ),

        index=False
    )

    print("\n" + "=" * 60)

    print("ALL SVM EXPERIMENTS COMPLETED")

    print("=" * 60)

    print("\nFinal comparison:")

    print(

        comparison[[

            "features",

            "accuracy",

            "balanced_accuracy",

            "precision",

            "recall",

            "f1_score",

            "roc_auc",

            "FP",

            "FN"

        ]]

        .round(4)

        .to_string(index=False)
    )

    print("\nComparison saved to:")

    print(
        os.path.abspath(
            os.path.join(
                args.output_dir,
                "comparison.csv"
            )
        )
    )


# ============================================================
# 8. Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(
            "Compare SVM using 32 and 55 ECG features."
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

        default="data/results/svm_30s"
    )

    parser.add_argument(

        "--C",

        type=float,

        default=1.0
    )

    parser.add_argument(

        "--cache_size",

        type=float,

        default=1024
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

    args = parser.parse_args()

    main(args)
