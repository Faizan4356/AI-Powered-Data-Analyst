"""Robust CSV loading: real-world CSVs vary in encoding and delimiter far
more than pandas' defaults assume. This tries a small, ordered set of
encodings and lets pandas auto-detect the delimiter within each, so a
semicolon-separated Windows-1252 export or a UTF-8-BOM file loads exactly
like a plain comma-separated UTF-8 one — no LLM involved, purely a parsing
concern.
"""
import io

import pandas as pd

# Ordered so the common case (plain UTF-8, or UTF-8 with a BOM) is tried
# first; later entries catch legacy/regional exports.
ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin-1", "utf-16"]


class CSVParseError(Exception):
    pass


def read_csv_robust(file_obj) -> pd.DataFrame:
    """Read an uploaded CSV file-like object of unknown encoding/delimiter."""
    raw = file_obj.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw.strip():
        raise CSVParseError("File is empty")

    attempts = []
    for encoding in ENCODINGS_TO_TRY:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError) as e:
            attempts.append(f"{encoding}: decode failed ({e})")
            continue

        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", skip_blank_lines=True)
        except Exception as e:
            attempts.append(f"{encoding}: parse failed ({e})")
            continue

        if df.shape[1] == 0:
            attempts.append(f"{encoding}: parsed to zero columns")
            continue

        return df

    raise CSVParseError(
        "Could not parse this CSV with any supported encoding/delimiter combination. "
        f"Tried: {'; '.join(attempts)}"
    )
