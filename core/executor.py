"""Deterministic execution engine. No LLM involvement.

Runs a validated OperationPlan against the real DataFrame using Pandas/DuckDB
and returns a structured, reproducible ExecutionResult. Every result carries
the exact code that produced it, so it can be independently re-run (audit log).
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import pandas as pd

from core.validation import OperationPlan, ValidationError, validate_sql


@dataclass
class ExecutionResult:
    op_type: str
    result_df: pd.DataFrame
    code_executed: str
    scalar_value: Optional[Any] = None
    error: Optional[str] = None


def execute_plan(plan: OperationPlan, df: pd.DataFrame) -> ExecutionResult:
    try:
        if plan.raw_sql:
            return _execute_sql(plan, df)
        return _execute_pandas(plan, df)
    except Exception as e:
        return ExecutionResult(op_type=plan.op_type, result_df=pd.DataFrame(), code_executed="", error=str(e))


def _apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    out = df
    for f in filters:
        col, op, val = f["column"], f["op"], f.get("value")
        series = out[col]
        if op == "==":
            out = out[series == val]
        elif op == "!=":
            out = out[series != val]
        elif op == ">":
            out = out[series > val]
        elif op == ">=":
            out = out[series >= val]
        elif op == "<":
            out = out[series < val]
        elif op == "<=":
            out = out[series <= val]
        elif op == "in":
            out = out[series.isin(val if isinstance(val, list) else [val])]
        elif op == "not in":
            out = out[~series.isin(val if isinstance(val, list) else [val])]
        elif op == "contains":
            out = out[series.astype(str).str.contains(str(val), case=False, na=False)]
    return out


def _execute_pandas(plan: OperationPlan, df: pd.DataFrame) -> ExecutionResult:
    code_lines = ["result = df"]
    working = _apply_filters(df, plan.filters)
    if plan.filters:
        code_lines.append(f"result = result[apply_filters({plan.filters!r})]")

    if plan.op_type == "describe":
        cols = plan.columns or list(working.columns)
        result_df = working[cols].describe(include="all").reset_index()
        code_lines.append(f"result = df[{cols!r}].describe(include='all')")

    elif plan.groupby and plan.agg_func and plan.agg_column:
        grouped = working.groupby(plan.groupby, dropna=False)[plan.agg_column].agg(plan.agg_func).reset_index()
        code_lines.append(
            f"result = df.groupby({plan.groupby!r})[{plan.agg_column!r}].agg({plan.agg_func!r}).reset_index()"
        )
        sort_col = plan.sort_by or plan.agg_column
        if sort_col in grouped.columns:
            grouped = grouped.sort_values(sort_col, ascending=not plan.sort_desc)
            code_lines.append(f"result = result.sort_values({sort_col!r}, ascending={not plan.sort_desc})")
        result_df = grouped

    elif plan.agg_func and plan.agg_column and not plan.groupby:
        value = getattr(working[plan.agg_column], plan.agg_func)()
        result_df = pd.DataFrame({plan.agg_column: [value]}, index=[plan.agg_func])
        code_lines.append(f"result = df[{plan.agg_column!r}].{plan.agg_func}()")
        if plan.limit:
            result_df = result_df.head(plan.limit)
        return ExecutionResult(
            op_type=plan.op_type, result_df=result_df, code_executed="\n".join(code_lines), scalar_value=value
        )

    else:
        cols = plan.columns or list(working.columns)
        result_df = working[cols]
        code_lines.append(f"result = df[{cols!r}]")

    if plan.sort_by and plan.sort_by in result_df.columns and not (plan.groupby and plan.agg_func):
        result_df = result_df.sort_values(plan.sort_by, ascending=not plan.sort_desc)
        code_lines.append(f"result = result.sort_values({plan.sort_by!r}, ascending={not plan.sort_desc})")

    if plan.limit:
        result_df = result_df.head(plan.limit)
        code_lines.append(f"result = result.head({plan.limit})")

    return ExecutionResult(op_type=plan.op_type, result_df=result_df, code_executed="\n".join(code_lines))


def _execute_sql(plan: OperationPlan, df: pd.DataFrame) -> ExecutionResult:
    sql = validate_sql(plan.raw_sql)
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    result_df = con.execute(sql).fetchdf()
    con.close()
    return ExecutionResult(op_type="raw_sql", result_df=result_df, code_executed=sql)
