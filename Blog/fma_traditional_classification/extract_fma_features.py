import json
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm

from feature_extractor import extract_feature

# config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = PROJECT_ROOT / "data" / "fma_metadata" / "tracks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fma_traditional_features" / "fma_small_features.csv"
SR = 22050
LIMIT = None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if LIMIT is not None:
        records = records[:LIMIT]

    for item in tqdm(records, desc="提取FMA-small传统特征"):
        track_id = item["track_id"]
        audio_path = Path(item["path"])

        try:
            y, sr = librosa.load(audio_path,sr=SR,mono=True,duration=None,)
            features = extract_feature(y, sr)
            row = {
                "track_id": track_id,
                "path": str(audio_path),
                "genre": item["genre"],
                "split": item["split"],
            }

            row.update(features)
            rows.append(row)

        except Exception as e:
            failed.append({
                "id": track_id,
                "path": str(audio_path),
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n特征提取完成")
    print(f"成功数量：{len(rows)}")
    print(f"失败数量：{len(failed)}")
    print(f"保存位置：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()