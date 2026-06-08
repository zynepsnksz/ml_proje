"""CSV yükleme ve train/test ayırma."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def validate_inputs(df: pd.DataFrame, raise_on_error: bool = True) -> bool:
    """Verilen özellik veri çerçevesindeki değerlerin fiziksel sınırlarını doğrular.
    
    Kurallar:
    - N, P, K >= 0
    - humidity: [0, 100]
    - ph: [0, 14]
    - rainfall >= 0
    """
    errors = []
    
    if "N" in df.columns and (df["N"] < 0).any():
        errors.append("N (Azot) değeri 0'dan küçük olamaz.")
        
    if "P" in df.columns and (df["P"] < 0).any():
        errors.append("P (Fosfor) değeri 0'dan küçük olamaz.")
        
    if "K" in df.columns and (df["K"] < 0).any():
        errors.append("K (Potasyum) değeri 0'dan küçük olamaz.")
        
    if "humidity" in df.columns and ((df["humidity"] < 0) | (df["humidity"] > 100)).any():
        errors.append("Nem (humidity) değeri 0 ile 100 arasında olmalıdır.")
        
    if "ph" in df.columns and ((df["ph"] < 0) | (df["ph"] > 14)).any():
        errors.append("pH değeri 0 ile 14 arasında olmalıdır.")
        
    if "rainfall" in df.columns and (df["rainfall"] < 0).any():
        errors.append("Yağış (rainfall) değeri 0'dan küçük olamaz.")
        
    if errors:
        error_msg = "Domain Doğrulama Hatası:\n" + "\n".join(errors)
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            print(f"Warning: {error_msg}")
            return False
            
    return True


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """CSV dosyasını okur ve doğrular."""
    df = pd.read_csv(path)
    validate_inputs(df, raise_on_error=True)
    return df


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
