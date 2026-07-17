from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_model(model_name: str):
    if model_name == "random":
        return DummyClassifier(
            strategy="uniform",
            random_state=42,
        )

    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            ),
        )
    if model_name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"未知模型：{model_name}")


def list_models():
    return [
        "random",
        "svm",
        "logistic",
        "random_forest",
    ]


if __name__ == "__main__":
    print("可用模型：")
    for name in list_models():
        print(name)