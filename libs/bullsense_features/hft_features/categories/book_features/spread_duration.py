import polars as pl
from hft_features.core.base import feature

"""
LOB_COLS = [
    'pb1','qb1','pb2','qb2','pb3','qb3','pb4','qb4','pb5','qb5',
    'pa1','qa1','pa2','qa2','pa3','qa3','pa4','qa4','pa5','qa5',
]
"""
from hft_features.core.base import feature
import polars as pl

@feature(deps={"mid_price"})
def financial_mid_duration(
    df: pl.DataFrame,
    datetime_col: str = "timestamp",
) -> pl.DataFrame:
    """
    mid_price'in değişmeden kaldığı süre (saniye).
    mid değiştikçe süre sıfırlanır, değişmediği sürece zaman farkları birikir.
    
    Eklenen sütun: mid_duration_s
    """
    # Eğer time_diff yoksa, saniye cinsinden hesapla
    if "time_diff" in df.columns:
        dt_s = pl.col("time_diff")
    else:
        dt_s = (
            pl.col(datetime_col)
              .diff()
              .dt.total_seconds()
              .fill_null(0.0)
              .alias("_dt_s")
        )
        df = df.with_columns(dt_s)
        dt_s = pl.col("_dt_s")

    mid = pl.col("mid_price")

    # mid değiştiğinde yeni streak başlat
    streak_id = (
        (mid != mid.shift(1))
        .cast(pl.Int64)
        .cumsum()
        .alias("_mid_streak_id")
    )
    df = df.with_columns(streak_id)

    # Her streak içinde dt_s'leri toplayarak "kaç saniyedir bu mid'deyiz" hesabı
    mid_duration = (
        dt_s.cumsum().over("_mid_streak_id")
            .alias("mid_duration_s")
    )

    out = df.with_columns(mid_duration)

    # Geçici kolonları temizle
    drop_cols = []
    if "_dt_s" in out.columns:
        drop_cols.append("_dt_s")
    drop_cols.append("_mid_streak_id")

    return out.drop(drop_cols)

@feature(deps={"financial_mid_duration"})
def mid_duration_log(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("mid_duration_s") + 1.0).log().alias("mid_duration_log")
    )
    


# @feature
# def add_distance_to_vwap(df: pl.DataFrame) -> pl.DataFrame:
#     """
#     VWAP’a uzaklık = (mid_price - vwap) / vwap
#     Eklenen sütun: dist_to_vwap
#     """
#     mid = (pl.col("pb1")+pl.col("pb1"))/2
#     dist = ((mid - pl.col("vwap"))/pl.col("vwap")).alias("dist_to_vwap")
#     return df.with_columns(dist)


@feature(deps={"financial_mid_duration"})
def spread_duration_features(df: pl.DataFrame, **params) -> pl.DataFrame:
    df = financial_mid_duration(df, **params)
    return df
