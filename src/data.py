from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

def load_ld2011(filename="LD2011_2014.txt") -> pd.DataFrame:
    """
    Читає UCI ELD2011-2014: перша колонка — datetime, далі 370 клієнтів (kW).
    ⚠️ У різних дзеркалах можуть відрізнятися десяткові роздільники.
       1) Пробуємо decimal="," (UCI оригінал).
       2) Якщо дані виглядають як object або NaN — пробуємо decimal=".".
    """
    fp = RAW / filename

    def _read(dec):
        return pd.read_csv(
            fp, sep=";", decimal=dec, parse_dates=[0], index_col=0, low_memory=False
        )

    # спроба з комою
    df = _read(",")
    # якщо колонки не числові — пробуємо крапку
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in df.dtypes):
        df = _read(".")

    df.index.name = "timestamp"
    df = df.sort_index()
    return df

def save_parquet_versions(df: pd.DataFrame) -> tuple[Path, Path]:
    """
    Зберігає версії 15-хв і годинну (mean) у Parquet.
    Якщо немає двигуна (pyarrow/fastparquet) — падаємо у CSV.
    """
    INTERIM.mkdir(parents=True, exist_ok=True)
    p15_parquet = INTERIM / "ld2011_15min.parquet"
    ph_parquet  = INTERIM / "ld2011_hourly.parquet"
    p15_csv     = INTERIM / "ld2011_15min.csv"
    ph_csv      = INTERIM / "ld2011_hourly.csv"

    try:
        # зберігаємо Parquet
        df.to_parquet(p15_parquet)
        df_hour = df.resample("1h").mean()
        df_hour.to_parquet(ph_parquet)
        return p15_parquet, ph_parquet
    except Exception as e:
        print("[WARN] Parquet недоступний, використовую CSV. Деталі:", e)
        df.to_csv(p15_csv)
        df_hour = df.resample("1H").mean()
        df_hour.to_csv(ph_csv)
        return p15_csv, ph_csv

if __name__ == "__main__":
    df = load_ld2011()
    p15, ph = save_parquet_versions(df)
    print(f"Saved 15-min: {p15}")
    print(f"Saved hourly: {ph}")
    print(f"Columns (series ids) sample: {list(df.columns)[:5]}")
