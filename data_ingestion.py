from __future__ import annotations
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
# import psycopg2
import re
 
import io
from utils.general_utils import timeit
from typing import Optional
 
import pandas as pd
import requests

import C5TC_ML_first_traded.config as config


_CAPES_SIGNAL_OCEAN_URLS = {
    # "c3_prepipe_ballasters": "https://app.signalocean.com/api/datalink/13b1994b-5044-4dc8-6851-08de8555c813?mode=Aggregate", # c3 prepipe (ballasters 5-15 days away from china) - old
    "c3_prepipe_ballasters": "https://app.signalocean.com/api/datalink/3b742943-4f7c-4982-407a-08deabe2ab73?mode=Aggregate",  # c3 prepipe (ballasters 5-15 days away from china) - new
    "c3_pipe_ballasters": "https://app.signalocean.com/api/datalink/c7842202-fe2d-443c-7041-08deabfa3fa1?mode=Aggregate",  # c3 pipe (includes indian ocean and west africa)
    "ballasters_atl": "https://app.signalocean.com/api/datalink/10218c8a-8b31-4870-407c-08deabe2ab73?mode=Aggregate",  # ballasters in Atlantic (e.g. for C8)
    "ballasters_pac": "https://app.signalocean.com/api/datalink/a1f5bf0b-95a1-42fd-407d-08deabe2ab73?mode=Aggregate",  # ballasters in Pacific (C10)
    "c3_laden_pipe": "https://app.signalocean.com/api/datalink/81e9703e-05a5-4fa3-7042-08deabfa3fa1?mode=Aggregate",  # c3 laden pipe (laden west africa / india heading to china)
    "laden_feast": "https://app.signalocean.com/api/datalink/a6b16a58-60cb-4123-407b-08deabe2ab73?mode=Aggregate",  # laden in FEAST (laden prepipe)
    # "laden_feast_no_china": "https://app.signalocean.com/api/datalink/a6b16a58-60cb-4123-407b-08deabe2ab73?mode=Aggregate",  # laden in FEAST but assume arriving vessels already priced in
    "discharge_congest": "https://app.signalocean.com/api/datalink/dbfeabf8-4d27-4a12-7043-08deabfa3fa1?mode=Aggregate",  # discharge congestion ww
    "load_congest": "https://app.signalocean.com/api/datalink/546531b6-54cc-40f1-407e-08deabe2ab73?mode=Aggregate",  # load congestion ww
    "brazil_exp": "https://app.signalocean.com/api/datalink/dbde0973-ea8e-4e61-7044-08deabfa3fa1?mode=Aggregate",  # smoothing to be applied
    "aus_exp": "https://app.signalocean.com/api/datalink/2aadd2d3-e66f-4c98-7045-08deabfa3fa1?mode=Aggregate",  # smoothing to be applied
    "fleet_size": "https://app.signalocean.com/api/datalink/e0ae1350-8672-4ab9-e31c-08deb033bc09?mode=Aggregate",
}


class SignalOceanAggregateLoader:
    def __init__(self, url: str):
        self.url = url
 
    def detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        for c in ("DayDate", "LoadDate", "Date"):
            if c in df.columns:
                return c
        return None
    
    def fetch(self) -> pd.DataFrame:
        resp = requests.get(self.url)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        date_col = self.detect_date_column(df) or "DayDate"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        return df

    def get_series(self) -> pd.Series:
        df = self.fetch()
        date_col = self.detect_date_column(df) or "DayDate"
        imo_cols = [c for c in df.columns if c.lower() == "imo"]
        if imo_cols:
            col = imo_cols[0]
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns found in {self.url}")
            col = numeric_cols[0]
        daily = df.groupby(date_col)[col].sum().reset_index()
        s = pd.Series(daily[col].values, index=pd.to_datetime(daily[date_col]))
        return s.dropna().sort_index()
 

@timeit
def load_commodities():
    ALL_SYMBOLS = config.FUEL_SYMBOLS | config.DRY_SYMBOLS | config.SEC_DRY_SYMBOLS

    query = text(f"""
        SELECT
            exchange_symbol,
            tenor,
            date,
            close,
            settlement,
            volume,
            CASE
                WHEN exchange_symbol IN ({sql_list(config.FUEL_SYMBOLS)})
                    THEN 'fuel_price'
                WHEN exchange_symbol IN ({sql_list(config.DRY_SYMBOLS)})
                    THEN 'dry_price'
                WHEN exchange_symbol IN ({sql_list(config.SEC_DRY_SYMBOLS)})
                    THEN 'sec_dry_price'
                ELSE 'other'
            END AS commodity_group
        FROM ectp.eod_prices
        WHERE exchange_symbol IN ({sql_list(ALL_SYMBOLS)})
        ORDER BY commodity_group, exchange_symbol, tenor, date
    """)

    engine = get_clickhouse_engine()
    with engine.begin() as conn:
        commodities_df = pd.read_sql(query, conn)
    commodities_df = (
        commodities_df.assign(date=lambda d: pd.to_datetime(d["date"]))
    )
    return commodities_df


def sql_list(values):
    return ", ".join(f"'{v}'" for v in values)


@timeit
def load_C5TC_spot():
    query = text("""
        SELECT tre.date, tre.px_last 
        FROM baltic.tb_routes_eod AS tre
        WHERE shortcode = 'C5TC'
        ORDER BY tre."date" ASC
    """)

    engine = get_itdaprod_engine()
    with engine.begin() as conn:
        C5TC_df = pd.read_sql(query, conn)

    C5TC_df = (
        C5TC_df
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .drop_duplicates(subset=["date"])
        .set_index("date")
        .sort_index()
        .squeeze("columns")
        .rename("C5TC_SPOT")
    )

    return C5TC_df


@timeit
def load_C5TC_all_tenors(start:int|None = None, end:int|None=None) -> pd.DataFrame:
    filters = ["ticker_identifier LIKE '5TC_C%'"]
    params = {}

    if start is not None and end is not None:
        filters.append("date BETWEEN :start_date AND :end_date")
        params["start_date"] = f"{start}-01-01"
        params["end_date"] = f"{end}-12-31"

    query = text(f"""
        SELECT date, period, value, ticker_identifier
        FROM baltic.tb_forward_curves
        WHERE {" AND ".join(filters)}
        ORDER BY period, date ASC
    """)

    engine = get_itdaprod_engine()
    with engine.begin() as conn:
        C5TC_nominal_df = pd.read_sql(query, conn, params=params)

    C5TC_nominal_df["date"] = pd.to_datetime(C5TC_nominal_df["date"])

    bad = C5TC_nominal_df.groupby(["date", "period"])["value"].transform("nunique").gt(1)
    if bad.any():
        print("Found duplicate date/period rows with different values. First values will be chosen")
        print(C5TC_nominal_df.loc[bad].sort_values(["date", "period"]))

    C5TC_nominal_df = (
        C5TC_nominal_df
        .drop_duplicates(["date", "period"], keep="first")
        .sort_values(["date", "period"])
        .reset_index(drop=True)
    )
    
    return C5TC_nominal_df


@timeit
def load_C5TC_per_tenor(tenor: str = "m1") -> pd.DataFrame:
    """
    EOD C5TC for given tenor
    """
    ticker_id = f"5TC_C{format_tenor_code(tenor)}"
    query = text("""
        SELECT date, value, period
        FROM baltic.tb_forward_curves
        WHERE ticker_identifier = :ticker
        ORDER BY date DESC
    """)
    engine = get_itdaprod_engine()
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker_id})


def format_tenor_code(s: str) -> str:
    """
    Converts tenor codes:
    m1 / M1 -> +1MON
    q2 / Q2 -> +2Q
    y1 / Y1 -> +1CAL
    """
    s = s.strip()
    match = re.fullmatch(r"([mMqQyY])(\d+)", s)
    if not match:
        raise ValueError(f"Invalid format: {s}. Expected like m1, q2, y1, etc.")
    typ, num_str = match.groups()
    num = int(num_str)
    if num <= 0:
        raise ValueError(f"Tenor index must be > 0, got {num}")
    mapping = {
        "M": "MON",
        "Q": "Q",
        "Y": "CAL"
    }
    unit = mapping[typ.upper()]
    return f"+{num}{unit}"


def get_itdaprod_engine():
    load_dotenv(config.ROOT / ".env")

    user = os.getenv("ITDAPROD_USER")
    password = os.getenv("ITDAPROD_PASSWORD")
    host = os.getenv("ITDAPROD_HOST")
    port = os.getenv("ITDAPROD_PORT")
    db = os.getenv("ITDAPROD_NAME")

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}",
        pool_pre_ping=True
    )


def get_clickhouse_engine():
    load_dotenv(config.ROOT / ".env")

    url = URL.create(
        "clickhousedb",
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=os.getenv("CLICKHOUSE_NAME", "default"),
    )

    return create_engine(url, pool_pre_ping=True)


def get_capes_data():
    return {
        name: SignalOceanAggregateLoader(url).get_series().sort_index()
        for name, url in _CAPES_SIGNAL_OCEAN_URLS.items()
    }
