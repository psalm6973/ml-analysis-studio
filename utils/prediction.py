import pandas as pd

from utils.model_persistence import load_model


def make_prediction(features):

    # Load saved model information
    model_data = load_model()

    pipeline = model_data["model_object"]
    target = model_data["target"]
    required_features = model_data["features"]

    # Check for missing features
    missing_features = [
        feature
        for feature in required_features
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # Keep only the features used during training
    input_data = {
        feature: features[feature]
        for feature in required_features
    }

    # Convert to one-row DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )

    # Make prediction
    prediction = pipeline.predict(
        input_df
    )

    # Convert NumPy value to normal Python value
    prediction_value = prediction[0]

    if hasattr(prediction_value, "item"):
        prediction_value = prediction_value.item()

    return {
        "target": target,
        "prediction": prediction_value
    }