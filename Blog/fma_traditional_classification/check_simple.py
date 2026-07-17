from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "fma_traditional_features"
    / "fma_small_features.csv"
)

META_COLS = ["track_id", "path", "genre", "split"]


def main():
    df = pd.read_csv(CSV_PATH)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
    ]

    print("========== 基本信息 ==========")
    print(f"样本数：{len(df)}")
    print(f"总列数：{len(df.columns)}")
    print(f"特征维度：{len(feature_cols)}")

    print("\n========== split 分布 ==========")
    print(df["split"].value_counts())

    print("\n========== genre 分布 ==========")
    print(df["genre"].value_counts())

    print("\n========== genre × split ==========")
    print(pd.crosstab(df["genre"], df["split"]))

    print("\n========== 重复 track_id ==========")
    print(df["track_id"].duplicated().sum())


    if len(feature_cols) != 87:
        print("\n警告：特征维度不是 87，请检查特征列。")
    else:
        print("\n检查通过：特征维度为 87。")


if __name__ == "__main__":
    main()