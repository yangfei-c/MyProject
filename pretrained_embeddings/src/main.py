from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "configs" / "config.yaml"

#屏蔽掉一些警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

from ast_pretrained import AstExtractor
from audio_process import load_audio, segment_audio
from clap_pretrained import ClapExtractor
from mert_pretrained import MertExtractor

def main() -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

#配置
    audio_path = (project_dir / config["audio"]["path"]).resolve()
    segment_seconds = int(config["audio"]["segment_seconds"])
    minimum_seconds = int(config["audio"]["minimum_seconds"])
    device_name = config["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output_root = (project_dir / config["runtime"]["output_dir"]).resolve()

#音频处理
    waveform, sample_rate = load_audio(audio_path)
    segments = segment_audio(waveform, sample_rate, segment_seconds, minimum_seconds)
    if not segments:
        raise RuntimeError("No valid audio segments were produced")

    track_dir = output_root / audio_path.stem
    track_dir.mkdir(parents=True, exist_ok=True)

#模型配置
    model_specs = [
        ("mert", MertExtractor),
        ("clap", ClapExtractor),
        ("ast", AstExtractor),
    ]

#开始提取
    for index, (model_name, Extractor) in enumerate(model_specs, start=1):
        model_cfg = config["models"][model_name]
        if not model_cfg.get("enabled", False):
            print(f"[{index}/3] {model_name}: skipped")
            continue

        print(f"[{index}/3] {model_name}: running")
        extractor = None
        try:
            extractor = Extractor(
                model_id=model_cfg["model_id"],
                device=device,
            )
            result = extractor.extract(segments=segments, original_sample_rate=sample_rate)

            model_dir = track_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            np.save(model_dir / "segment_embeddings.npy", result["segment_embeddings"])
            np.save(model_dir / "track_embedding.npy", result["track_embedding"])
            print(f"[{index}/3] {model_name}: done, dim={result['embedding_dim']}")
        except Exception as exc:
            print(f"[{index}/3] {model_name}: failed - {exc}")
        finally:
            del extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Saved to: {track_dir}")


if __name__ == "__main__":
    main()
