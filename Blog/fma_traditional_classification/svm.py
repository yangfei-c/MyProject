from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_CSV = (
    PROJECT_ROOT
    / "data"
    / "fma_traditional_features"
    / "fma_small_features.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "fma_traditional_results"
)

META_COLS = ["track_id", "path", "genre", "split"]


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_CSV)

    feature_cols = [
        col for col in df.columns
        if col not in META_COLS
    ]

    print("========== 数据信息 ==========")
    print(f"样本数：{len(df)}")
    print(f"特征维度：{len(feature_cols)}")
    print(df["split"].value_counts())

    train_df = df[df["split"] == "training"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    x_train = train_df[feature_cols]
    y_train = train_df["genre"]

    x_val = val_df[feature_cols]
    y_val = val_df["genre"]

    x_test = test_df[feature_cols]
    y_test = test_df["genre"]

    labels = sorted(df["genre"].unique())

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        ),
    )

    print("\n========== 开始训练 LinearSVC ==========")
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print("\n========== 验证集结果 ==========")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro-F1: {val_f1:.4f}")

    print("\n========== 测试集结果 ==========")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")

    report = classification_report(
        y_test,
        test_pred,
        labels=labels,
        digits=4,
        zero_division=0,
    )

    print("\n========== 测试集分类报告 ==========")
    print(report)

    report_path = RESULT_DIR / "svm_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_path = RESULT_DIR / "svm_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {
            "model": "LinearSVC",
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }
    ])

    summary_path = RESULT_DIR / "svm_result.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== 保存完成 ==========")
    print(f"结果摘要：{summary_path}")
    print(f"分类报告：{report_path}")
    print(f"混淆矩阵：{cm_path}")


if __name__ == "__main__":
    main()