import subprocess
import sys
from pathlib import Path

from fma_models import list_models

CURRENT_DIR = Path(__file__).resolve().parent

TRAIN_SCRIPT = CURRENT_DIR / "train_model.py"

def main():
    models = list_models()

    print("========== 准备运行全部模型 ==========")
    print("模型列表：")
    for model_name in models:
        print(f"- {model_name}")

    for model_name in models:
        print("\n" + "=" * 60)
        print(f"开始运行模型：{model_name}")
        print("=" * 60)

        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            model_name,
        ]

        subprocess.run(
            command,
            check=True,
        )

    print("\n========== 全部模型运行完成 ==========")


if __name__ == "__main__":
    main()