import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)

from utils.preprocessing import (
    preprocess_data,
    build_preprocessor
)


# Maximum number of features shown to the user
MAX_PREDICTION_FEATURES = 5


def get_problem_type(df, target_column):

    target = df[target_column]

    unique_values = target.nunique(
        dropna=True
    )

    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "classification"

    elif pd.api.types.is_numeric_dtype(target):

        if unique_values <= 10:
            return "classification"

        return "regression"

    return "classification"


def create_model(model_name):

    if model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=1000
        )

    elif model_name == "knn":

        return KNeighborsClassifier(
            n_neighbors=5
        )

    elif model_name == "svm":

        return SVC()

    elif model_name == "decision_tree":

        return DecisionTreeClassifier(
            random_state=42
        )

    elif model_name == "linear_regression":

        return LinearRegression()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


def calculate_feature_importance(
    X,
    y,
    problem_type
):
    """
    Calculate an importance score for each
    original feature.

    This allows us to select a small number
    of useful features for the prediction form.
    """

    importance_scores = {}

    # Prepare target
    if problem_type == "classification":

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        ):

            y_encoded = pd.Series(
                pd.factorize(y)[0],
                index=y.index
            )

        else:

            y_encoded = y

    else:

        y_encoded = y

    for column in X.columns:

        data = X[column]

        try:

            # Categorical feature
            if (
                pd.api.types.is_object_dtype(data)
                or pd.api.types.is_string_dtype(data)
                or pd.api.types.is_categorical_dtype(data)
                or pd.api.types.is_bool_dtype(data)
            ):

                # Fill missing values
                data = data.fillna(
                    "Missing"
                )

                # Convert categories to numbers
                encoded_data = pd.Series(
                    pd.factorize(data)[0],
                    index=data.index
                )

                X_feature = (
                    encoded_data
                    .to_numpy()
                    .reshape(-1, 1)
                )

                if problem_type == "classification":

                    score = mutual_info_classif(
                        X_feature,
                        y_encoded,
                        discrete_features=True,
                        random_state=42
                    )[0]

                else:

                    score = mutual_info_regression(
                        X_feature,
                        y_encoded,
                        discrete_features=True,
                        random_state=42
                    )[0]

            # Numerical feature
            else:

                numeric_data = pd.to_numeric(
                    data,
                    errors="coerce"
                )

                numeric_data = numeric_data.fillna(
                    numeric_data.median()
                )

                X_feature = (
                    numeric_data
                    .to_numpy()
                    .reshape(-1, 1)
                )

                if problem_type == "classification":

                    score = mutual_info_classif(
                        X_feature,
                        y_encoded,
                        discrete_features=False,
                        random_state=42
                    )[0]

                else:

                    score = mutual_info_regression(
                        X_feature,
                        y_encoded,
                        discrete_features=False,
                        random_state=42
                    )[0]

            importance_scores[column] = float(
                score
            )

        except Exception:

            # If a feature causes an issue,
            # simply give it zero importance.
            importance_scores[column] = 0.0

    return importance_scores


def make_feature_schema(X):

    feature_schema = []

    for column in X.columns:

        data = X[column]

        # Numerical feature
        if pd.api.types.is_numeric_dtype(
            data
        ):

            clean_data = data.dropna()

            if len(clean_data) > 0:

                minimum = clean_data.min()
                maximum = clean_data.max()
                default = clean_data.median()

                if hasattr(minimum, "item"):
                    minimum = minimum.item()

                if hasattr(maximum, "item"):
                    maximum = maximum.item()

                if hasattr(default, "item"):
                    default = default.item()

            else:

                minimum = None
                maximum = None
                default = None

            feature_schema.append({
                "name": column,
                "type": "number",
                "min": minimum,
                "max": maximum,
                "default": default
            })

        # Categorical feature
        else:

            values = (
                data
                .dropna()
                .unique()
                .tolist()
            )

            clean_values = []

            for value in values:

                if hasattr(
                    value,
                    "item"
                ):
                    value = value.item()

                clean_values.append(
                    value
                )

            clean_values.sort(
                key=lambda value: str(value)
            )

            feature_schema.append({
                "name": column,
                "type": "category",
                "options": clean_values
            })

    return feature_schema


def train_model(
    df,
    target_column,
    model_name
):

    # Determine problem type
    problem_type = get_problem_type(
        df,
        target_column
    )

    # Prepare initial features
    X, y, _ = preprocess_data(
        df,
        target_column
    )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable features were found."
        )

    # Allowed models
    classification_models = [
        "logistic_regression",
        "knn",
        "svm",
        "decision_tree"
    ]

    regression_models = [
        "linear_regression"
    ]

    # Check model compatibility
    if (
        problem_type == "classification"
        and model_name not in classification_models
    ):

        raise ValueError(
            "Selected model is not suitable "
            "for classification."
        )

    if (
        problem_type == "regression"
        and model_name not in regression_models
    ):

        raise ValueError(
            "Selected model is not suitable "
            "for regression."
        )

    # ------------------------------------------------
    # Split data
    # ------------------------------------------------

    if problem_type == "classification":

        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )
            )

        except ValueError:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

    else:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

    # ------------------------------------------------
    # Find important features
    # ------------------------------------------------

    importance_scores = calculate_feature_importance(
        X_train,
        y_train,
        problem_type
    )

    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Select top 5
    selected_features = [
        feature
        for feature, score
        in sorted_features[
            :MAX_PREDICTION_FEATURES
        ]
    ]

    # Preserve original dataset order
    selected_features = [
        column
        for column in X.columns
        if column in selected_features
    ]

    # ------------------------------------------------
    # Train final model using selected features
    # ------------------------------------------------

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]

    final_preprocessor = build_preprocessor(
        X_train_selected
    )

    model = create_model(
        model_name
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            final_preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train_selected,
        y_train
    )

    # ------------------------------------------------
    # Test model
    # ------------------------------------------------

    predictions = pipeline.predict(
        X_test_selected
    )

    # ------------------------------------------------
    # Classification metrics
    # ------------------------------------------------

    if problem_type == "classification":

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        metrics = {
            "accuracy": round(
                accuracy * 100,
                2
            ),
            "precision": round(
                precision * 100,
                2
            ),
            "recall": round(
                recall * 100,
                2
            ),
            "f1_score": round(
                f1 * 100,
                2
            )
        }

        reliability_score = metrics[
            "f1_score"
        ]

        if reliability_score >= 70:

            reliability = "reliable"

        else:

            reliability = "needs_improvement"

    # ------------------------------------------------
    # Regression metrics
    # ------------------------------------------------

    else:

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        metrics = {
            "mae": round(
                mae,
                4
            ),
            "mse": round(
                mse,
                4
            ),
            "rmse": round(
                rmse,
                4
            ),
            "r2_score": round(
                r2,
                4
            )
        }

        reliability_score = metrics[
            "r2_score"
        ]

        if reliability_score >= 0.30:

            reliability = "reliable"

        else:

            reliability = "needs_improvement"

    # ------------------------------------------------
    # Feature types
    # ------------------------------------------------

    feature_types = {}

    for column in selected_features:

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            feature_types[column] = "number"

        else:

            feature_types[column] = "category"

    # ------------------------------------------------
    # Feature schema
    # ------------------------------------------------

    feature_schema = make_feature_schema(
        X_train_selected
    )

    # ------------------------------------------------
    # Return result
    # ------------------------------------------------

    return {
        "problem_type": problem_type,
        "target": target_column,
        "model": model_name,
        "metrics": metrics,
        "reliability": reliability,
        "reliability_score": reliability_score,
        "features": selected_features,
        "feature_types": feature_types,
        "feature_schema": feature_schema,
        "model_object": pipeline
    }