import pandas as pd
import re


def is_id_column(column, data=None):

    name = column.lower().strip()

    id_names = [
        "id",
        "index",
        "identifier",
        "record_id",
        "customer_id",
        "user_id",
        "employee_id",
        "student_id",
        "product_id",
        "passenger_id"
    ]

    if name in id_names:
        return True

    if name.endswith("_id"):
        return True

    if name.endswith("id") and len(name) > 2:
        return True

    if "identifier" in name:
        return True

    return False


def calculate_target_score(
    df,
    column,
    column_position
):

    data = df[column]

    name = column.lower().strip()

    unique_values = data.nunique(
        dropna=True
    )

    total_rows = len(data)

    if total_rows == 0:
        return -100

    if unique_values <= 1:
        return -100

    # Ignore obvious IDs
    if is_id_column(column, data):
        return -100

    score = 0

    # ==================================================
    # CATEGORICAL COLUMNS
    # ==================================================

    if (
        pd.api.types.is_object_dtype(data)
        or pd.api.types.is_string_dtype(data)
        or pd.api.types.is_categorical_dtype(data)
        or pd.api.types.is_bool_dtype(data)
    ):

        # Small number of categories
        if 2 <= unique_values <= 20:
            score += 40

        # Binary target
        if unique_values == 2:
            score += 10

        # Exact target-like names
        strong_target_names = [
            "target",
            "label",
            "class",
            "species",
            "survived",
            "outcome",
            "result",
            "status",
            "churn",
            "default",
            "approved",
            "fraud",
            "diagnosis",
            "grade",
            "score",
            "mark",
            "marks"
        ]

        # IMPORTANT:
        # Use exact name matching here.
        # This prevents "Pstatus" from being
        # incorrectly treated as "status".
        if name in strong_target_names:
            score += 60

        # Check common target-like endings
        if (
            name.endswith("_target")
            or name.endswith("_label")
            or name.endswith("_score")
            or name.endswith("_result")
            or name.endswith("_outcome")
        ):
            score += 40

        # Columns near the end of a dataset
        position_ratio = (
            column_position /
            max(len(df.columns) - 1, 1)
        )

        if position_ratio >= 0.8:
            score += 10

        return score

    # ==================================================
    # NUMERICAL COLUMNS
    # ==================================================

    if pd.api.types.is_numeric_dtype(data):

        # Regression-like numerical target
        if unique_values > 10:
            score += 30

        # Small/medium numerical target
        elif 2 <= unique_values <= 10:
            score += 15

        # Strong numerical target names
        strong_numeric_names = [
            "target",
            "label",
            "score",
            "price",
            "salary",
            "sales",
            "revenue",
            "profit",
            "amount",
            "value",
            "survived",
            "outcome",
            "result",
            "rating",
            "grade",
            "grades",
            "mark",
            "marks",
            "final"
        ]

        # Exact name matching
        if name in strong_numeric_names:
            score += 60

        # Common target-like endings
        if (
            name.endswith("_target")
            or name.endswith("_score")
            or name.endswith("_result")
            or name.endswith("_value")
        ):
            score += 40

        # Recognize G1, G2, G3, etc.
        # This is especially useful for datasets
        # containing sequential grades.
        if re.fullmatch(
            r"g\d+",
            name
        ):
            score += 80

        # Columns near the end
        position_ratio = (
            column_position /
            max(len(df.columns) - 1, 1)
        )

        if position_ratio >= 0.8:
            score += 15

        # Last column bonus
        if column_position == len(df.columns) - 1:
            score += 15

        return score

    return -100


def suggest_targets(df):

    scored_targets = []

    for position, column in enumerate(
        df.columns
    ):

        score = calculate_target_score(
            df,
            column,
            position
        )

        if score >= 0:

            scored_targets.append(
                (
                    column,
                    score
                )
            )

    scored_targets.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [
        column
        for column, score
        in scored_targets[:3]
    ]


def determine_problem_type(
    df,
    target_column
):

    target = df[target_column]

    unique_values = target.nunique(
        dropna=True
    )

    # Categorical target
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "classification"

    # Numerical target
    if pd.api.types.is_numeric_dtype(target):

        if unique_values <= 10:
            return "classification"

        return "regression"

    return "classification"


def get_available_models(
    problem_type
):

    if problem_type == "classification":

        return [
            {
                "name": "Logistic Regression",
                "value": "logistic_regression"
            },
            {
                "name": "K-Nearest Neighbors",
                "value": "knn"
            },
            {
                "name": "Support Vector Machine",
                "value": "svm"
            },
            {
                "name": "Decision Tree",
                "value": "decision_tree"
            }
        ]

    if problem_type == "regression":

        return [
            {
                "name": "Linear Regression",
                "value": "linear_regression"
            }
        ]

    return []