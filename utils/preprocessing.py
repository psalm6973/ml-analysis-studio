import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def is_id_column(column):
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


def is_high_cardinality_column(column, data):

    # Only check categorical columns
    if not (
        pd.api.types.is_object_dtype(data)
        or pd.api.types.is_string_dtype(data)
        or pd.api.types.is_categorical_dtype(data)
        or pd.api.types.is_bool_dtype(data)
    ):
        return False

    unique_values = data.nunique(
        dropna=True
    )

    # More than 20 different categories
    # is considered too many for our
    # prediction form.
    return unique_values > 20


def build_preprocessor(X):

    # Numerical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns

    # Categorical columns
    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool"
        ]
    ).columns

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return preprocessor


def preprocess_data(df, target_column):

    # Separate features and target
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Remove ID columns
    id_columns = [
        column
        for column in X.columns
        if is_id_column(column)
    ]

    if id_columns:
        X = X.drop(
            columns=id_columns
        )

    # Remove high-cardinality categorical columns
    high_cardinality_columns = [
        column
        for column in X.columns
        if is_high_cardinality_column(
            column,
            X[column]
        )
    ]

    if high_cardinality_columns:
        X = X.drop(
            columns=high_cardinality_columns
        )

    # Create the preprocessing pipeline
    preprocessor = build_preprocessor(
        X
    )

    # IMPORTANT:
    # We do not fit the preprocessor here.
    # It will be fitted only on training data.

    return X, y, preprocessor