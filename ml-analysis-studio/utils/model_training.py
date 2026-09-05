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

from utils.preprocessing import preprocess_data


# =========================================================
# DETERMINE PROBLEM TYPE
# =========================================================

def get_problem_type(df, target_column):

    target = df[target_column]

    unique_values = target.nunique(dropna=True)

    # Categorical target
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):

        return "classification"

    # Numerical target
    elif pd.api.types.is_numeric_dtype(target):

        if unique_values <= 10:
            return "classification"

        return "regression"

    return "classification"


# =========================================================
# CREATE MODEL
# =========================================================

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


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(df, target_column, model_name):

    # -----------------------------------------------------
    # Determine problem type
    # -----------------------------------------------------

    problem_type = get_problem_type(
        df,
        target_column
    )


    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # -----------------------------------------------------
    # Create preprocessing + model
    # -----------------------------------------------------

    # We create the preprocessing pipeline using the
    # existing preprocessing function.

    _, _, preprocessor = preprocess_data(
        df,
        target_column
    )


    model = create_model(
        model_name
    )


    # IMPORTANT:
    #
    # The preprocessor and model are combined into
    # ONE pipeline.
    #
    # This allows us to later give raw user input
    # directly to the trained pipeline.

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])


    # -----------------------------------------------------
    # Split RAW data
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # -----------------------------------------------------
    # Train pipeline
    # -----------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # Predict test data
    # -----------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================

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


    # =====================================================
    # REGRESSION METRICS
    # =====================================================

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


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "problem_type": problem_type,

        "target": target_column,

        "model": model_name,

        "metrics": metrics,

        # THIS is the important new part.
        #
        # It contains both preprocessing + trained model.

        "model_object": pipeline,

        # Keep the preprocessor separately too.
        # It may be useful later.

        "preprocessor": preprocessor

    }