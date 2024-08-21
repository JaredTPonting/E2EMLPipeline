"""
Load raw data
Handle missing values
Encode cat variables
Scale numerical features if necessary
Save processed data
"""
from typing import Tuple, Any

import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(file_path: str = "../data/raw/train.csv") -> pd.DataFrame:
    """
    Laod titanic dataset from the specific path.

    :param file_path: Path to raw titanic dataset
    :return: pd.DataFrame: Loaded dataframe
    """

    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> tuple[DataFrame, pd.Series]:
    """
    Pre processing the dataset

    :param df: unprocessed data
    :return: processed data
    """

    unused_cols = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
    to_remove = [col for col in df.columns if col in unused_cols]

    y = df['Survived']
    x = df.drop(to_remove, axis=1)

    numeric_features = ["Age", "Fare"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_features = ["Pclass", "Sex", "Embarked"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_preprocessed = preprocessor.fit_transform(x)
    return pd.DataFrame(X_preprocessed.tolist()), y


def save_data(X: pd.DataFrame, y: pd.Series, X_path: str = "../data/processed/X.csv", y_path: str = "../data/processed/y.csv") -> None:
    """
    Save the preprocessed features and labels to the given paths

    :param X: Preprocessed features
    :param y: Labels
    :param X_path: Path to save Features
    :param y_path: Path to save Labels
    """
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)


if __name__ == '__main__':
    raw_df = load_data()
    X, y = preprocess_data(raw_df)
    save_data(X,y)