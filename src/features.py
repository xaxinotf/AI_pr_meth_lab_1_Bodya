import pandas as pd

def add_calendar_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["dow"] = index.dayofweek
    cal["month"] = index.month
    return cal

def make_lags(y: pd.Series, lags=(1,2,3,24,25,168)) -> pd.DataFrame:
    return pd.concat({f"lag_{l}": y.shift(l) for l in lags}, axis=1)

def build_feature_table(df_wide: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """
    df_wide: індекс — datetime, колонки — клієнти (наприклад MT_001, …)
    Повертає таблицю з лагами + календарними ознаками і ціллю 'y'.
    """
    y = df_wide[series_id].astype(float)
    X_lag = make_lags(y)
    X_cal = add_calendar_index(df_wide.index)
    X = pd.concat([X_lag, X_cal], axis=1)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    return data
