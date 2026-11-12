
import os
import glob
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)


def main(args):

    # ==================================================
    # 1. 讀取所有受試者的 CSV
    # ==================================================

    files = sorted(
        glob.glob(
            os.path.join(
                args.input_dir,
                "[0-9][0-9][0-9][0-9][0-9].csv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found: {args.input_dir}"
        )

    print("CSV files found:", len(files))

    data_list = []

    for i, file in enumerate(files, start=1):

        df = pd.read_csv(file)

        df["participant_id"] = os.path.splitext(
            os.path.basename(file)
        )[0]

        # 保留每個視窗在原始 CSV 中的列位置
        # 方便日後追溯原始資料
        df["source_row"] = np.arange(len(df))

        data_list.append(df)

        if i % 10 == 0 or i == len(files):
            print(
                f"Loaded {i}/{len(files)} participants"
            )

    data = pd.concat(
        data_list,
        ignore_index=True
    )

    del data_list

    print("\nTotal windows:", len(data))
    print(
        "Participants:",
        data["participant_id"].nunique()
    )

    # ==================================================
    # 2. 建立 Stress / Non-stress 標籤
    # ==================================================

    # Label 0:
    # baseline + low/moderate/high physical activity
    #
    # Label 1:
    # mental_stress

    if "category" not in data.columns:
        raise ValueError(
            "Input CSV must contain a 'category' column."
        )

    data["category"] = (
        data["category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # 支援兩種常見活動類別名稱
    # 統一成 low_activity 等名稱
    category_mapping = {
        "baseline": "baseline",
        "mental_stress": "mental_stress",

        "low_activity": "low_activity",
        "low_physical_activity": "low_activity",

        "moderate_activity": "moderate_activity",
        "moderate_physical_activity": "moderate_activity",

        "high_activity": "high_activity",
        "high_physical_activity": "high_activity"
    }

    data["category"] = data["category"].map(
        category_mapping
    )

    valid_categories = [
        "mental_stress",
        "baseline",
        "low_activity",
        "moderate_activity",
        "high_activity"
    ]

    before_filter = len(data)

    data = data[
        data["category"].isin(valid_categories)
    ].copy()

    print(
        "\nWindows excluded due to unknown category:",
        before_filter - len(data)
    )

    if data.empty:
        raise ValueError(
            "No valid windows remain after category filtering."
        )

    data["stress_label"] = (
        data["category"] == "mental_stress"
    ).astype(int)

    print("\nCategory distribution:")
    print(data["category"].value_counts())

    print("\nClass distribution:")
    print(data["stress_label"].value_counts())

    # ==================================================
    # 3. 取得 ECG 特徵
    # ==================================================

    exclude_columns = [
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
        "source_row",
        "start",
        "end",
        "start_time",
        "end_time",
        "Unnamed: 0"
    ]

    X = data.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    # 只保留數值型特徵
    X = X.select_dtypes(
        include=[np.number]
    )

    X = X.loc[
        :,
        ~X.columns.str.startswith("Unnamed:")
    ]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 移除所有視窗都沒有值的特徵
    # 若發生此情況，直接報錯，避免32/55項不一致
    all_missing_features = X.columns[
        X.isna().all()
    ].tolist()

    if all_missing_features:
        raise ValueError(
            "Features containing only missing values: "
            + str(all_missing_features)
        )

    feature_names = X.columns.tolist()

    y = data["stress_label"]
    groups = data["participant_id"]

    print("\nFeature count:", X.shape[1])
    print("Feature names:")
    print(feature_names)

    if X.shape[1] != args.feature_count:
        raise ValueError(
            f"Expected {args.feature_count} features, "
            f"found {X.shape[1]}. "
            "Check input columns and feature extraction."
        )

    # ==================================================
    # 4. 按照受試者切分 Train / Test
    # ==================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_subjects = groups.iloc[train_idx]
    test_subjects = groups.iloc[test_idx]

    print(
        "\nTraining participants:",
        train_subjects.nunique()
    )

    print(
        "Testing participants:",
        test_subjects.nunique()
    )

    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))

    if (
        y_train.nunique() != 2
        or y_test.nunique() != 2
    ):
        raise ValueError(
            "Train and test must contain both classes."
        )

    assert not set(train_subjects).intersection(
        set(test_subjects)
    ), "Subject leakage detected!"

    # ==================================================
    # 5. 處理缺失值
    # ==================================================

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    # ==================================================
    # 6. 建立 Random Forest
    # ==================================================

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # ==================================================
    # 7. 訓練模型
    # ==================================================

    print("\nTraining Random Forest...")

    model.fit(
        X_train_imputed,
        y_train
    )

    print("Training completed.")

    # ==================================================
    # 8. 預測每個測試視窗
    # ==================================================

    y_pred = model.predict(
        X_test_imputed
    )

    y_prob = model.predict_proba(
        X_test_imputed
    )[:, 1]

    # ==================================================
    # 9. 評估整體模型
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print("\n===== Random Forest Results =====")

    print("Accuracy:", accuracy)
    print("F1-score:", f1)
    print("ROC-AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("FPR:", fpr)

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
            zero_division=0
        )
    )

    # ==================================================
    # 10. 建立輸出資料夾
    # ==================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ==================================================
    # 11. 儲存整體結果 results.csv
    # ==================================================

    results = pd.DataFrame([{
        "features": args.feature_count,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "train_participants": train_subjects.nunique(),
        "test_participants": test_subjects.nunique()
    }])

    results.to_csv(
        os.path.join(
            args.output_dir,
            "results.csv"
        ),
        index=False
    )

    # ==================================================
    # 12. 儲存每一筆測試視窗 predictions.csv
    # ==================================================

    # 從原始資料取出測試集
    # 保留原本的 ECG 特徵數值
    predictions = data.iloc[
        test_idx
    ].copy()

    # 保留原始 label（如果存在）
    # 但另以 stress_label 作為本次二元分類真值
    predictions = predictions.rename(
        columns={
            "stress_label": "true_label"
        }
    )

    predictions["predicted_label"] = y_pred

    predictions["stress_probability"] = y_prob

    predictions["prediction_type"] = np.select(
        [
            (
                (predictions["true_label"] == 0)
                & (predictions["predicted_label"] == 0)
            ),
            (
                (predictions["true_label"] == 0)
                & (predictions["predicted_label"] == 1)
            ),
            (
                (predictions["true_label"] == 1)
                & (predictions["predicted_label"] == 0)
            ),
            (
                (predictions["true_label"] == 1)
                & (predictions["predicted_label"] == 1)
            )
        ],
        [
            "TN",
            "FP",
            "FN",
            "TP"
        ],
        default="Unknown"
    )

    # 重新整理欄位順序
    metadata_columns = [
        "participant_id",
        "source_row",
        "category",
        "true_label",
        "predicted_label",
        "stress_probability",
        "prediction_type"
    ]

    other_columns = [
        col for col in predictions.columns
        if col not in metadata_columns
    ]

    predictions = predictions[
        metadata_columns + other_columns
    ]

    predictions.to_csv(
        os.path.join(
            args.output_dir,
            "predictions.csv"
        ),
        index=False
    )

    print(
        "\nSaved predictions:",
        len(predictions)
    )

    # ==================================================
    # 13. 儲存 FP / TN / FN / TP 各組資料
    # ==================================================

    for prediction_type in [
        "FP",
        "TN",
        "FN",
        "TP"
    ]:

        subset = predictions[
            predictions["prediction_type"]
            == prediction_type
        ].copy()

        filename = (
            prediction_type.lower()
            + "_windows.csv"
        )

        subset.to_csv(
            os.path.join(
                args.output_dir,
                filename
            ),
            index=False
        )

        print(
            f"{prediction_type} windows:",
            len(subset)
        )

    # 另外提供容易辨識的 FP 檔名
    false_positives = predictions[
        predictions["prediction_type"] == "FP"
    ].copy()

    false_positives.to_csv(
        os.path.join(
            args.output_dir,
            "false_positives.csv"
        ),
        index=False
    )

    # ==================================================
    # 14. 分析不同活動情境的預測結果
    # ==================================================

    category_results = (
        predictions
        .groupby(
            ["category", "prediction_type"],
            observed=True
        )
        .size()
        .unstack(fill_value=0)
    )

    for col in ["TN", "FP", "FN", "TP"]:
        if col not in category_results.columns:
            category_results[col] = 0

    category_results = category_results[
        ["TN", "FP", "FN", "TP"]
    ]

    category_results["total"] = (
        category_results[
            ["TN", "FP", "FN", "TP"]
        ].sum(axis=1)
    )

    # FP rate 只對真實 Label 0 的視窗有意義
    category_results["fp_rate"] = np.where(
        (
            category_results["TN"]
            + category_results["FP"]
        ) > 0,
        category_results["FP"]
        / (
            category_results["TN"]
            + category_results["FP"]
        ).replace(0, np.nan),
        np.nan
    )

    category_results = (
        category_results.reset_index()
    )

    category_results.to_csv(
        os.path.join(
            args.output_dir,
            "category_results.csv"
        ),
        index=False
    )

    print("\n===== Results by Category =====")
    print(category_results.to_string(index=False))

    # ==================================================
    # 15. 儲存特徵重要性
    # ==================================================

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.insert(
        0,
        "rank",
        np.arange(1, len(importance) + 1)
    )

    importance.to_csv(
        os.path.join(
            args.output_dir,
            "feature_importance.csv"
        ),
        index=False
    )

    # ==================================================
    # 16. 儲存 Train / Test 受試者名單
    # ==================================================

    split_records = []

    for subject in sorted(set(train_subjects)):
        split_records.append({
            "participant_id": subject,
            "split": "train"
        })

    for subject in sorted(set(test_subjects)):
        split_records.append({
            "participant_id": subject,
            "split": "test"
        })

    pd.DataFrame(
        split_records
    ).to_csv(
        os.path.join(
            args.output_dir,
            "subject_split.csv"
        ),
        index=False
    )

    # ==================================================
    # 17. 最終檢查
    # ==================================================

    assert len(predictions) == len(y_test)

    assert (
        len(false_positives) == fp
    ), "FP count mismatch!"

    print("\n================================")
    print("All results saved successfully.")
    print("Output directory:", args.output_dir)
    print("================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing participant feature CSV files"
    )

    parser.add_argument(
        "--feature_count",
        type=int,
        choices=[32, 55],
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    main(args)
