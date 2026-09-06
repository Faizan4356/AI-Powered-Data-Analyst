"""Multi-file/multi-table support: infer join keys, let the LLM choose which
joins to run, then execute the merge deterministically.

Key inference is pure Pandas — name similarity plus actual value-overlap
between columns — precisely so the LLM is never asked to guess whether two
columns are really joinable from names alone. It only picks among candidates
that have already been verified to share real values.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from core.llm_client import LLMClient, LLMUnavailableError, safe_json_parse
from core.validation import JoinStep, ValidationError, validate_join_plan

MIN_OVERLAP_RATIO = 0.3


@dataclass
class JoinCandidate:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    overlap_ratio: float
    name_similarity: float


@dataclass
class JoinResult:
    merged_df: pd.DataFrame
    steps_applied: list
    code_executed: str
    error: Optional[str] = None


def _normalize_col(name: str) -> str:
    n = name.lower()
    n = re.sub(r"(^|_)(id|key|code)$", "", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n or name.lower()


def infer_join_candidates(tables: dict) -> list:
    """tables: {table_name: DataFrame}. Returns candidates scored by real value
    overlap between column pairs across different tables — no LLM involved."""
    candidates = []
    names = list(tables.keys())

    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left_df, right_df = tables[left_name], tables[right_name]
            for lcol in left_df.columns:
                for rcol in right_df.columns:
                    norm_l, norm_r = _normalize_col(lcol), _normalize_col(rcol)
                    name_sim = 1.0 if norm_l == norm_r else (0.5 if norm_l in norm_r or norm_r in norm_l else 0.0)
                    if name_sim == 0.0:
                        continue

                    left_vals = set(left_df[lcol].dropna().unique().tolist())
                    right_vals = set(right_df[rcol].dropna().unique().tolist())
                    if not left_vals or not right_vals:
                        continue

                    overlap = len(left_vals & right_vals)
                    smaller = min(len(left_vals), len(right_vals))
                    overlap_ratio = overlap / smaller if smaller else 0.0

                    if overlap_ratio >= MIN_OVERLAP_RATIO:
                        candidates.append(
                            JoinCandidate(
                                left_table=left_name,
                                left_column=lcol,
                                right_table=right_name,
                                right_column=rcol,
                                overlap_ratio=round(overlap_ratio, 3),
                                name_similarity=name_sim,
                            )
                        )

    candidates.sort(key=lambda c: (c.overlap_ratio, c.name_similarity), reverse=True)
    return candidates


def compute_pairwise_overlap_matrix(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """Every left-column x right-column value-overlap ratio between two tables,
    including pairs below the auto-join threshold — a full transparency view of
    which dimensions are (and aren't) joinable, not just the ones picked as
    candidates. Pure Pandas, no LLM, no name-similarity filtering."""
    matrix = pd.DataFrame(index=left_df.columns, columns=right_df.columns, dtype=float)

    for lcol in left_df.columns:
        left_vals = set(left_df[lcol].dropna().unique().tolist())
        for rcol in right_df.columns:
            right_vals = set(right_df[rcol].dropna().unique().tolist())
            if not left_vals or not right_vals:
                matrix.loc[lcol, rcol] = 0.0
                continue
            overlap = len(left_vals & right_vals)
            smaller = min(len(left_vals), len(right_vals))
            matrix.loc[lcol, rcol] = round(overlap / smaller, 3) if smaller else 0.0

    return matrix


SYSTEM_PROMPT = """You are a join planner for a data analysis tool. You are given
several tables' schemas and a list of CANDIDATE join keys that have already been
verified to share real overlapping values between the two columns. You must only
choose joins from this candidate list — never invent a table, column, or key that
is not listed. Pick the minimal set of joins needed to connect all the tables into
one analyzable table.

Respond with strict JSON only:
{"joins": [{"left_table": "...", "right_table": "...", "left_key": "...", "right_key": "...", "how": "inner|left|right|outer"}]}
"""


def plan_joins(tables: dict, candidates: list, llm: LLMClient) -> list:
    if not candidates:
        raise ValidationError("No candidate join keys were found between the uploaded tables")

    schema_lines = []
    for name, df in tables.items():
        schema_lines.append(f"Table '{name}': columns = {list(df.columns)}, rows = {len(df)}")

    candidate_lines = [
        f"- {c.left_table}.{c.left_column} <-> {c.right_table}.{c.right_column} (value overlap ratio: {c.overlap_ratio})"
        for c in candidates
    ]

    user_prompt = "\n".join(schema_lines) + "\n\nCandidate join keys:\n" + "\n".join(candidate_lines)

    try:
        resp = llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        plan_dict = safe_json_parse(resp.text)
    except (LLMUnavailableError, ValueError) as e:
        logger.warning(f"Join planning degraded, using top candidate only: {e}")
        top = candidates[0]
        plan_dict = {
            "joins": [
                {
                    "left_table": top.left_table,
                    "right_table": top.right_table,
                    "left_key": top.left_column,
                    "right_key": top.right_column,
                    "how": "inner",
                }
            ]
        }

    try:
        return validate_join_plan(plan_dict, tables, candidates=candidates)
    except ValidationError as e:
        logger.warning(f"LLM join plan failed validation ({e}); falling back to top candidate")
        top = candidates[0]
        fallback = {
            "joins": [
                {
                    "left_table": top.left_table,
                    "right_table": top.right_table,
                    "left_key": top.left_column,
                    "right_key": top.right_column,
                    "how": "inner",
                }
            ]
        }
        return validate_join_plan(fallback, tables, candidates=candidates)


def execute_joins(tables: dict, steps: list) -> JoinResult:
    try:
        code_lines = []
        merged_name = steps[0].left_table
        merged = tables[merged_name]
        merged_tables = {merged_name}

        for step in steps:
            if step.left_table in merged_tables:
                base, base_key = merged, step.left_key
                other_name, other_key = step.right_table, step.right_key
            elif step.right_table in merged_tables:
                base, base_key = merged, step.right_key
                other_name, other_key = step.left_table, step.left_key
            else:
                raise ValidationError(f"Join step references tables not yet connected: {step}")

            other = tables[other_name]
            merged = base.merge(
                other, left_on=base_key, right_on=other_key, how=step.how, suffixes=("", f"_{other_name}")
            )
            merged_tables.add(other_name)
            code_lines.append(
                f"merged = merged.merge(tables['{other_name}'], left_on='{base_key}', right_on='{other_key}', how='{step.how}')"
            )

        return JoinResult(merged_df=merged, steps_applied=steps, code_executed="\n".join(code_lines))
    except Exception as e:
        return JoinResult(merged_df=pd.DataFrame(), steps_applied=[], code_executed="", error=str(e))
