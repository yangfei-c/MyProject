from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = (PROJECT_ROOT/ "data"/ "fma_traditional/features"/ "fma_small_features.csv")

META_COLS = ["track_id","path","genre","split",]


def load_fma_dataset():
    df = pd.read_csv(FEATURE_CSV)

    feature_cols = [
        col for col in df.columns
        if col not in META_COLS
    ]

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

    return {
        "df": df,
        "feature_cols": feature_cols,
        "labels": labels,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    data = load_fma_dataset()

    df = data["df"]
    feature_cols = data["feature_cols"]

    print("========== FMA 特征数据集 ==========")
    print(f"样本数：{len(df)}")
    print(f"特征维度：{len(feature_cols)}")

    print("\n========== split 分布 ==========")
    print(df["split"].value_counts())

    print("\n========== genre 分布 ==========")
    print(df["genre"].value_counts())

    print("\n========== labels ==========")
    print(data["labels"])