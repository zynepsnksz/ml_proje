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
