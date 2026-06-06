"""CSV yükleme ve train/test ayırma."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """CSV dosyasını okur."""
    return pd.read_csv(path)


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Özellik ve hedef sütunlarını ayırır."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split uygular."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
