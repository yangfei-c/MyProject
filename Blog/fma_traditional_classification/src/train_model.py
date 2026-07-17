import argparse
from pathlib import Path

from fma_dataset import load_fma_dataset
from fma_models import build_model, list_models
from fma_evaluate import evaluate_


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (PROJECT_ROOT/ "data"/ "fma_traditional"/ "results")


def parse_args():
    parser = argparse.ArgumentParser(description="FMAm Classification.")
    parser.add_argument("--model",type=str,default="svm",choices=list_models(),help="选择要训练的模型",)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    print(f"========== 开始训练模型：{model_name} ==========")

    RESULT_DIR.mkdir(parents=True,exist_ok=True,)

    data = load_fma_dataset()

    x_train = data["x_train"]
    y_train = data["y_train"]

    x_val = data["x_val"]
    y_val = data["y_val"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    labels = data["labels"]

    model = build_model(model_name)

    model.fit(x_train,y_train,)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    evaluate_(model_name=model_name,labels=labels,y_val=y_val,val_pred=val_pred,y_test=y_test,test_pred=test_pred,result_root=RESULT_DIR,)

if __name__ == "__main__":
    main()