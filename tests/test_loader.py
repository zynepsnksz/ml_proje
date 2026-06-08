"""Veri yükleme ve split testleri."""

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.data.loader import get_feature_target, load_data, split_data


def test_load_data_shape():
    df = load_data()
    assert df.shape == (2200, 8)


def test_get_feature_target_columns():
    df = load_data()
    X, y = get_feature_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN


def test_split_data_stratified():
    df = load_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - TEST_SIZE) < 0.01

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    assert train_counts.min() == 80
    assert test_counts.min() == 20


def test_validate_inputs():
    from src.data.loader import validate_inputs
    import pytest
    
    # Valid input DataFrame
    valid_df = pd.DataFrame([{
        "N": 50, "P": 40, "K": 30, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0
    }])
    assert validate_inputs(valid_df) is True
    
    # Invalid N < 0
    invalid_n = pd.DataFrame([{
        "N": -10, "P": 40, "K": 30, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0
    }])
    with pytest.raises(ValueError, match="Azot"):
        validate_inputs(invalid_n)
        
    # Invalid pH > 14
    invalid_ph = pd.DataFrame([{
        "N": 50, "P": 40, "K": 30, "temperature": 25.0, "humidity": 80.0, "ph": 15.0, "rainfall": 200.0
    }])
    with pytest.raises(ValueError, match="pH"):
        validate_inputs(invalid_ph)
