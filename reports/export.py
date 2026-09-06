"""One-click exportable reports: EDA + Q&A session + charts as HTML.

HTML is used instead of PDF to avoid a heavyweight rendering dependency;
the exported file is self-contained and opens in any browser. Nothing here
computes new numbers — it only formats what the executor already produced.
"""
from datetime import datetime

import pandas as pd


def build_html_report(eda_narrative: list[str], chat_history: list[dict], profile_summary: dict) -> str:
    generated_at = datetime.utcnow().isoformat() + "Z"

    narrative_html = "".join(f"<li>{n}</li>" for n in eda_narrative)
    chat_html = "".join(
        f"<div class='turn'><p><b>Q:</b> {t['question']}</p><p><b>A:</b> {t['answer']}</p></div>"
        for t in chat_history
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI Data Analyst Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; color: #1a1a1a; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
.turn {{ background: #f6f6f6; padding: 10px; border-radius: 6px; margin-bottom: 10px; }}
.meta {{ color: #666; font-size: 0.9em; }}
</style></head>
<body>
<h1>AI Data Analyst Report</h1>
<p class="meta">Generated {generated_at}</p>

<h2>Dataset summary</h2>
<p>{profile_summary.get('row_count', '?')} rows, {profile_summary.get('column_count', '?')} columns,
{profile_summary.get('duplicate_rows', '?')} duplicate rows.</p>

<h2>EDA narrative</h2>
<ul>{narrative_html}</ul>

<h2>Q&A session</h2>
{chat_html}
</body></html>"""


def save_report(html: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
