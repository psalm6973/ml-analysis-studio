import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(df, target_column):

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]


    # Find numerical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns


    # Find categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns


    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
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
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])


    # Combine both pipelines
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

        ]
    )


    # Transform features
    X_processed = preprocessor.fit_transform(X)


    return X_processed, y, preprocessor

