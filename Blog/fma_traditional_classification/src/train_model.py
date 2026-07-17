from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from fma_dataset import load_fma_dataset
from fma_models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "fma_traditional"
    / "results"
)

# "random","svm","logistic","random_forest",
MODEL_NAME = "svm"


def main():
    global result_dir
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_fma_dataset()

    x_train = data["x_train"]
    y_train = data["y_train"]

    x_val = data["x_val"]
    y_val = data["y_val"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    labels = data["labels"]

    model = build_model(MODEL_NAME)

    print(f"========== 开始训练模型：{MODEL_NAME} ==========")

    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

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
    result_dir=RESULT_DIR / f"{MODEL_NAME}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir/"classification_report.txt"
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

    cm_path = result_dir/"confusion_matrix.csv"
    cm_df.to_csv(
        cm_path,
        encoding="utf-8-sig",
    )

    result_df = pd.DataFrame([
        {
            "model": MODEL_NAME,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }
    ])

    result_path = result_dir/"result.csv"
    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 保存完成 ==========")
    print(f"结果摘要：{result_path}")
    print(f"分类报告：{report_path}")
    print(f"混淆矩阵：{cm_path}")


if __name__ == "__main__":
    main()