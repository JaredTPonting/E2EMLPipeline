import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model_training import train_model, evaluate_model


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    return X, y


def test_train_model(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10]}
    best_model, best_params = train_model(X, y, model, param_grid)
    assert best_model is not None
    assert 'C' in best_params


def test_evaluate_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    accuracy, precision, recall, f1, report = evaluate_model(model, X, y)
    assert accuracy > 0
    assert precision > 0
    assert recall > 0
    assert f1 > 0
    assert isinstance(report, str)
