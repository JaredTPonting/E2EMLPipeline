import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib


def load_data(X_path='../data/processed/X.csv', y_path='../data/processed/y.csv'):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y


def train_model(X, y, model, parma_grid):
    grid_search = GridSearchCV(model, parma_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    }

    best_models = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        best_model, best_params = train_model(X_train, y_train, model, param_grids[model_name])
        best_models[model_name] = best_model
        print(f'Best parameters for {model_name}: {best_params}')

        accuracy, precision, recall, f1, report = evaluate_model(best_model, X_test, y_test)
        print(f"Evaluation metrics for {model_name}:\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Classification Report: \n {report}")

        joblib.dump(best_model, f'../models/{model_name}.pkl')


if __name__ == '__main__':
    main()