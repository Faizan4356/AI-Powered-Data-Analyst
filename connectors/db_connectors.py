"""Live data connectors (Postgres/MySQL/Google Sheets). Phase 2 — stubbed.

Kept separate from core/ so adding a connector never touches the
LLM-as-planner / code-as-executor pipeline: a connector's only job is to
produce a pandas.DataFrame that the existing pipeline can already consume.
"""
import pandas as pd


def read_sql_database(connection_string: str, query: str) -> pd.DataFrame:
    """Read a table/query from a SQL database via SQLAlchemy.

    NOTE: `query` must be a trusted, developer-supplied SELECT — never pass
    LLM- or user-composed SQL here directly; route it through
    core.validation.validate_sql first, same as the DuckDB path in executor.py.
    """
    from sqlalchemy import create_engine

    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)


def read_google_sheet(sheet_url: str) -> pd.DataFrame:
    """Read a public/shared Google Sheet as CSV export."""
    if "/edit" in sheet_url:
        sheet_url = sheet_url.split("/edit")[0]
    csv_url = f"{sheet_url}/export?format=csv"
    return pd.read_csv(csv_url)
