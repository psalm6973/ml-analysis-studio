import os
import joblib


MODEL_PATH = "models/model.joblib"


def save_model(model_data):
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model_data,
        MODEL_PATH
    )


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. "
            "Please train a model first."
        )

    return joblib.load(
        MODEL_PATH
    )