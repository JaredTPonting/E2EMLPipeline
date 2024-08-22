"""
Load raw data
Handle missing values
Encode cat variables
Scale numerical features if necessary
Save processed data
"""
from typing import Tuple, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess_data(input_path: str = "../data/raw/train.csv", output_X_path: str = "../data/processed/X.csv",
                    output_y_path: str = "../data/processed/y.csv"):
    # Load the dataset with original column names
    df = pd.read_csv(input_path)

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Identify numerical and categorical columns
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    # Define transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)

    # Convert the result back to a DataFrame with original column names
    # Get the column names after one-hot encoding
    ohe_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)

    # Combine numeric and one-hot encoded column names
    all_columns = numeric_features + list(ohe_columns)

    # Create a DataFrame with the new column names
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)

    # Save the preprocessed features and target
    X_preprocessed_df.to_csv(output_X_path, index=False)
    y.to_csv(output_y_path, index=False)


if __name__ == "__main__":
    preprocess_data()
