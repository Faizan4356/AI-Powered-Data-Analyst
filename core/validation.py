"""Input sanitization and operation-plan validation.

Nothing in operation_planner or executor should trust LLM output directly.
Every plan is validated against the actual DataFrame schema here before
any code runs against it.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

ALLOWED_AGG_FUNCS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std", "var"}
ALLOWED_FILTER_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains"}
ALLOWED_OP_TYPES = {"aggregation", "trend", "comparison", "groupby", "filter", "raw_sql", "describe"}
ALLOWED_JOIN_TYPES = {"inner", "left", "right", "outer"}

# Blocks DDL/DML and multi-statement injection in any user- or LLM-generated SQL.
_SQL_FORBIDDEN = re.compile(
    r"\b(drop|delete|insert|update|alter|create|attach|detach|pragma|copy|exec|grant)\b",
    re.IGNORECASE,
)
_SQL_MULTISTATEMENT = re.compile(r";\s*\S")


class ValidationError(Exception):
    pass


@dataclass
class OperationPlan:
    op_type: str
    columns: list = field(default_factory=list)
    filters: list = field(default_factory=list)  # [{"column", "op", "value"}]
    groupby: list = field(default_factory=list)
    agg_func: Optional[str] = None
    agg_column: Optional[str] = None
    sort_by: Optional[str] = None
    sort_desc: bool = True
    limit: Optional[int] = None
    raw_sql: Optional[str] = None


def validate_plan(plan_dict: dict, df: pd.DataFrame) -> OperationPlan:
    op_type = plan_dict.get("op_type")
    if op_type not in ALLOWED_OP_TYPES:
        raise ValidationError(f"Unsupported op_type: {op_type}")

    columns = plan_dict.get("columns") or []
    _validate_columns(columns, df)

    filters = plan_dict.get("filters") or []
    for f in filters:
        if f.get("column") not in df.columns:
            raise ValidationError(f"Unknown filter column: {f.get('column')}")
        if f.get("op") not in ALLOWED_FILTER_OPS:
            raise ValidationError(f"Unsupported filter operator: {f.get('op')}")

    groupby = plan_dict.get("groupby") or []
    _validate_columns(groupby, df)

    agg_func = plan_dict.get("agg_func")
    if agg_func and agg_func not in ALLOWED_AGG_FUNCS:
        raise ValidationError(f"Unsupported aggregation function: {agg_func}")

    agg_column = plan_dict.get("agg_column")
    if agg_column and agg_column not in df.columns:
        raise ValidationError(f"Unknown aggregation column: {agg_column}")

    sort_by = plan_dict.get("sort_by")
    if sort_by and sort_by not in df.columns and sort_by != agg_column:
        raise ValidationError(f"Unknown sort column: {sort_by}")

    raw_sql = plan_dict.get("raw_sql")
    if raw_sql:
        validate_sql(raw_sql)

    limit = plan_dict.get("limit")
    if limit is not None:
        limit = min(int(limit), 10000)

    return OperationPlan(
        op_type=op_type,
        columns=columns,
        filters=filters,
        groupby=groupby,
        agg_func=agg_func,
        agg_column=agg_column,
        sort_by=sort_by,
        sort_desc=bool(plan_dict.get("sort_desc", True)),
        limit=limit,
        raw_sql=raw_sql,
    )


def _validate_columns(columns: list, df: pd.DataFrame):
    for c in columns:
        if c not in df.columns:
            raise ValidationError(f"Unknown column: {c}")


def validate_sql(sql: str) -> str:
    """Guard against destructive/multi-statement SQL. Read-only SELECTs only."""
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE):
        raise ValidationError("Only SELECT/WITH queries are allowed")
    if _SQL_FORBIDDEN.search(stripped):
        raise ValidationError("Query contains a forbidden keyword")
    if _SQL_MULTISTATEMENT.search(stripped):
        raise ValidationError("Multi-statement queries are not allowed")
    return stripped


@dataclass
class JoinStep:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    how: str


def validate_join_plan(plan_dict: dict, tables: dict, candidates: Optional[list] = None) -> list:
    """Validate an LLM-proposed join plan against the actual uploaded tables.

    `tables` maps table_name -> DataFrame. Every table/column reference is
    checked against real schemas before any merge runs. If `candidates` is
    given (a list of objects with left_table/left_column/right_table/right_column,
    e.g. core.join_planner.JoinCandidate), each proposed join must match one of
    them — the LLM may only pick among key pairs already verified to share real
    overlapping values, never invent an arbitrary column pair.
    """
    steps_in = plan_dict.get("joins") or []
    if not steps_in:
        raise ValidationError("Join plan contains no joins")

    steps = []
    for s in steps_in:
        left_table, right_table = s.get("left_table"), s.get("right_table")
        left_key, right_key = s.get("left_key"), s.get("right_key")
        how = s.get("how", "inner")

        if left_table not in tables:
            raise ValidationError(f"Unknown left_table: {left_table}")
        if right_table not in tables:
            raise ValidationError(f"Unknown right_table: {right_table}")
        if left_key not in tables[left_table].columns:
            raise ValidationError(f"Unknown left_key '{left_key}' in table '{left_table}'")
        if right_key not in tables[right_table].columns:
            raise ValidationError(f"Unknown right_key '{right_key}' in table '{right_table}'")
        if how not in ALLOWED_JOIN_TYPES:
            raise ValidationError(f"Unsupported join type: {how}")

        if candidates is not None:
            forward = (left_table, left_key, right_table, right_key)
            backward = (right_table, right_key, left_table, left_key)
            is_candidate = any(
                (c.left_table, c.left_column, c.right_table, c.right_column) in (forward, backward) for c in candidates
            )
            if not is_candidate:
                raise ValidationError(
                    f"Join on {left_table}.{left_key} <-> {right_table}.{right_key} is not a verified candidate key"
                )

        steps.append(JoinStep(left_table=left_table, right_table=right_table, left_key=left_key, right_key=right_key, how=how))

    return steps


def sanitize_column_name(name: str) -> str:
    """Normalize an uploaded column name to a safe SQL/pandas identifier."""
    clean = re.sub(r"\W+", "_", str(name).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "col"
    if clean[0].isdigit():
        clean = f"c_{clean}"
    return clean
