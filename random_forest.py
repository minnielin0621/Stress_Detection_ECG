
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

    # 1. 讀取所有受試者的 CSV
    files = sorted(glob.glob(
        os.path.join(
            args.input_dir,
            "[0-9][0-9][0-9][0-9][0-9].csv"
        )
    ))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found: {args.input_dir}"
        )

    data_list = []

    for file in files:

        df = pd.read_csv(file)

        df["participant_id"] = os.path.splitext(
            os.path.basename(file)
        )[0]

        data_list.append(df)

    data = pd.concat(
        data_list,
        ignore_index=True
    )

    print("Total windows:", len(data))
    print("Participants:", data["participant_id"].nunique())


    # 2. 建立 Stress / Non-stress 標籤
    # mental_stress = 1
    # baseline + physical activity = 0

    data["category"] = (
        data["category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    valid_categories = [
        "mental_stress",
        "baseline",
        "low_physical_activity",
        "moderate_physical_activity",
        "high_physical_activity"
    ]

    data = data[
        data["category"].isin(valid_categories)
    ].copy()

    data["stress_label"] = (
        data["category"] == "mental_stress"
    ).astype(int)

    print("\nClass distribution:")
    print(data["stress_label"].value_counts())


    # 3. 取得 ECG 特徵
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

    X = X.select_dtypes(
        include=[np.number]
    )

    X = X.loc[
        :, ~X.columns.str.startswith("Unnamed:")
    ]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = data["stress_label"]

    groups = data["participant_id"]

    print("\nFeature count:", X.shape[1])

    if X.shape[1] != args.feature_count:
        raise ValueError(
            f"Expected {args.feature_count} features, "
            f"found {X.shape[1]}"
        )


    # 4. 按照受試者分成 Train / Test
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

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_subjects = groups.iloc[train_idx]
    test_subjects = groups.iloc[test_idx]

    print("\nTraining participants:",
          train_subjects.nunique())

    print("Testing participants:",
          test_subjects.nunique())

    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Train and test must contain both classes."
        )

    assert not set(train_subjects).intersection(
        set(test_subjects)
    )


    # 5. 處理缺失值
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)


    # 6. 建立 Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    # 7. 訓練模型
    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # 8. 預測測試資料
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # 9. 評估模型
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
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

    print("\n===== Results =====")

    print("Accuracy:", accuracy)
    print("F1-score:", f1)
    print("ROC-AUC:", auc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))


    # 10. 儲存結果
    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    results = pd.DataFrame([{
        "features": args.feature_count,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }])

    results.to_csv(
        os.path.join(
            args.output_dir,
            "results.csv"
        ),
        index=False
    )

    print("\nResults saved to:")
    print(args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True
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
