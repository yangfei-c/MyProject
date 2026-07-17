from pathlib import Path

import pandas as pd

from sklearn.metrics import (accuracy_score,f1_score,classification_report,confusion_matrix,)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true,y_pred,average="macro",zero_division=0,)
    return accuracy, macro_f1


def evaluate_(model_name, labels, y_val, val_pred, y_test, test_pred, result_root, ):
    result_dir = Path(result_root) / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    val_accuracy, val_macro_f1 = compute_metrics(
        y_true=y_val,
        y_pred=val_pred,
    )
    test_accuracy, test_macro_f1 = compute_metrics(
        y_true=y_test,
        y_pred=test_pred,
    )

    print("\n========== 验证集结果 ==========")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro-F1: {val_macro_f1:.4f}")

    print("\n========== 测试集结果 ==========")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")

    result_df = pd.DataFrame([
        {
            "model": model_name,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
        }
    ])

    result_path = result_dir / "result.csv"
    result_df.to_csv(result_path,index=False,encoding="utf-8-sig",)
    report = classification_report(y_test,test_pred,labels=labels,digits=4,zero_division=0,)

    print("\n========== 测试集分类报告 ==========")
    print(report)
    report_path = result_dir / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    #混淆矩阵
    cm = confusion_matrix(y_test,test_pred,labels=labels,)
    cm_df = pd.DataFrame(cm,index=labels,columns=labels,)
    cm_path = result_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path,encoding="utf-8-sig",)

    print("\n========== 保存完成 ==========")
    print(f"结果摘要：{result_path}")
    print(f"分类报告：{report_path}")
    print(f"混淆矩阵：{cm_path}")