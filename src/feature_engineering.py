import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crop recommendation veri seti için yeni özellikler üretir.
    Orijinal veri setini bozmamak için kopya üzerinde çalışır.
    """

    df = df.copy()

    # Toplam toprak besin miktarı
    df["NPK_Total"] = df["N"] + df["P"] + df["K"]

    # Besin oranları
    df["N_to_P"] = df["N"] / (df["P"] + 1)
    df["N_to_K"] = df["N"] / (df["K"] + 1)
    df["P_to_K"] = df["P"] / (df["K"] + 1)

    # Basit toprak verimlilik indeksi
    df["Soil_Fertility"] = (
        0.4 * df["N"] +
        0.3 * df["P"] +
        0.3 * df["K"]
    )

    return df


def get_engineered_feature_columns() -> list:
    """
    Üretilen yeni özelliklerin isimlerini döndürür.
    """

    return [
        "NPK_Total",
        "N_to_P",
        "N_to_K",
        "P_to_K",
        "Soil_Fertility"
    ]