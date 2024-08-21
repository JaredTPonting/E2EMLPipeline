import pytest
import pandas as pd
from src.preprocessing import preprocess_data, load_data


@pytest.fixture
def sample_data():
    data = {
        "Survived": [0, 1],
        "Pclass": [3, 1],
        "Sex": ["male", "female"],
        "Age": [22, None],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Fare": [7.25, 71.2833],
        "Embarked": ["S", "C"],
    }
    return pd.DataFrame(data)


def test_load_data():
    """Test loading of data."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_preprocess_data(sample_data):
    """Test the preprocessing of data."""
    X, y = preprocess_data(sample_data)
    assert X.shape[1] > 0  # Ensure there are features
    assert len(X) == len(y)  # Ensure rows match between features and labels
    assert not X.isnull().values.any()  # Ensure no missing values after preprocessing
