import pandas as pd


# =========================================================
# CHECK IF COLUMN IS AN ID
# =========================================================

def is_id_column(column, data):

    name = column.lower().strip()

    # -----------------------------------------------------
    # Strong ID names
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # ID suffix
    # -----------------------------------------------------

    if name.endswith("_id"):
        return True


    # Examples:
    # PassengerId
    # CustomerId
    # UserId

    if name.endswith("id") and len(name) > 2:
        return True


    if "identifier" in name:
        return True


    # -----------------------------------------------------
    # DO NOT reject a column just because it has
    # many unique values.
    #
    # Price, Area, Salary, Revenue etc. can naturally
    # have many unique values.
    # -----------------------------------------------------

    return False


# =========================================================
# TARGET SCORE
# =========================================================

def calculate_target_score(df, column):

    data = df[column]

    name = column.lower().strip()

    unique_values = data.nunique(dropna=True)

    total_rows = len(data)


    # -----------------------------------------------------
    # Invalid columns
    # -----------------------------------------------------

    if total_rows == 0:
        return -100


    if unique_values <= 1:
        return -100


    # -----------------------------------------------------
    # Ignore ID columns
    # -----------------------------------------------------

    if is_id_column(column, data):
        return -100


    score = 0


    # =====================================================
    # CATEGORICAL TARGET
    # =====================================================

    if (
        pd.api.types.is_object_dtype(data)
        or pd.api.types.is_string_dtype(data)
        or pd.api.types.is_categorical_dtype(data)
        or pd.api.types.is_bool_dtype(data)
    ):

        # Reasonable number of classes
        if 2 <= unique_values <= 20:

            score += 40


        # Binary classification
        if unique_values == 2:

            score += 10


        # Names that strongly suggest a target
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
            "diagnosis"

        ]


        for word in strong_target_names:

            if word in name:

                score += 40
                break


        return score


    # =====================================================
    # NUMERICAL TARGET
    # =====================================================

    if pd.api.types.is_numeric_dtype(data):

        # -------------------------------------------------
        # Continuous numerical columns
        # -------------------------------------------------

        if unique_values > 10:

            score += 30


        # -------------------------------------------------
        # Numerical columns with few values
        # may represent classes
        # -------------------------------------------------

        elif 2 <= unique_values <= 10:

            score += 15


        # -------------------------------------------------
        # Strong target names
        # -------------------------------------------------

        strong_target_names = [

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
            "rating"

        ]


        for word in strong_target_names:

            if word in name:

                score += 40
                break


        return score


    return -100


# =========================================================
# SUGGEST TARGETS
# =========================================================

def suggest_targets(df):

    scored_targets = []


    for column in df.columns:

        score = calculate_target_score(
            df,
            column
        )


        if score >= 0:

            scored_targets.append(
                (column, score)
            )


    # -----------------------------------------------------
    # Highest confidence first
    # -----------------------------------------------------

    scored_targets.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # -----------------------------------------------------
    # Return maximum 3 targets
    # -----------------------------------------------------

    return [
        column
        for column, score
        in scored_targets[:3]
    ]


# =========================================================
# DETERMINE PROBLEM TYPE
# =========================================================

def determine_problem_type(df, target_column):

    target = df[target_column]

    unique_values = target.nunique(
        dropna=True
    )


    # -----------------------------------------------------
    # CATEGORICAL / TEXT
    # -----------------------------------------------------

    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):

        return "classification"


    # -----------------------------------------------------
    # NUMERICAL
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(target):

        # Small number of unique values
        # can represent classes.
        if unique_values <= 10:

            return "classification"


        return "regression"


    return "classification"


# =========================================================
# AVAILABLE MODELS
# =========================================================

def get_available_models(problem_type):

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    if problem_type == "regression":

        return [

            {
                "name": "Linear Regression",
                "value": "linear_regression"
            }

        ]


    return []

