import pandas as pd

from utils.model_persistence import load_model


def make_prediction(features):

    # Load saved model
    model_data = load_model()

    pipeline = model_data["model_object"]
    target = model_data["target"]

    required_features = model_data[
        "features"
    ]

    feature_schema = model_data[
        "feature_schema"
    ]

    # Create quick schema lookup
    schema_lookup = {
        item["name"]: item
        for item in feature_schema
    }

    # Check missing features
    missing_features = [
        feature
        for feature in required_features
        if feature not in features
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(
                missing_features
            )
        )

    validated_features = {}

    # Validate every feature
    for feature in required_features:

        value = features[feature]

        schema = schema_lookup.get(
            feature
        )

        if schema is None:
            continue

        # ==========================================
        # NUMBER
        # ==========================================

        if schema["type"] == "number":

            try:

                number_value = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"{feature} must be a valid number."
                )

            minimum = schema.get(
                "min"
            )

            maximum = schema.get(
                "max"
            )

            # Only apply min/max when available
            if (
                minimum is not None
                and number_value < minimum
            ):

                raise ValueError(
                    f"{feature} must be at least "
                    f"{minimum}."
                )

            if (
                maximum is not None
                and number_value > maximum
            ):

                raise ValueError(
                    f"{feature} must be at most "
                    f"{maximum}."
                )

            # If it's a dropdown with numeric values,
            # make sure the chosen value is allowed.
            if (
                schema.get("input_type")
                == "select"
            ):

                options = schema.get(
                    "options",
                    []
                )

                numeric_options = []

                for option in options:

                    try:

                        numeric_options.append(
                            float(option)
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        pass

                if (
                    number_value
                    not in numeric_options
                ):

                    raise ValueError(
                        f"Invalid value for {feature}."
                    )

                # Keep it as an integer when appropriate
                if number_value.is_integer():

                    number_value = int(
                        number_value
                    )

            validated_features[
                feature
            ] = number_value

        # ==========================================
        # CATEGORY
        # ==========================================

        else:

            options = schema.get(
                "options",
                []
            )

            if value not in options:

                raise ValueError(
                    f"Invalid value for {feature}. "
                    f"Choose one of: "
                    + ", ".join(
                        str(option)
                        for option in options
                    )
                )

            validated_features[
                feature
            ] = value

    # Create one-row DataFrame
    input_df = pd.DataFrame(
        [validated_features]
    )

    # Make prediction
    prediction = pipeline.predict(
        input_df
    )

    prediction_value = prediction[0]

    # Convert NumPy value to Python value
    if hasattr(
        prediction_value,
        "item"
    ):

        prediction_value = (
            prediction_value.item()
        )

    return {
        "target": target,
        "prediction": prediction_value
    }