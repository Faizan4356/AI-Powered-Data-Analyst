"""AI Data Analyst — Streamlit entrypoint.

Architecture: question -> intent_parser (LLM) -> operation_planner (LLM,
validated JSON) -> executor (deterministic Pandas/SQL, NO LLM) ->
response_composer (LLM, given only the computed result). See core/ for
each stage. The LLM never performs or invents calculations.

UI: a horizontal navbar at the top of the page selects one section at a
time — the sidebar is reserved for data upload and LLM status only.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

pio.templates.default = "plotly_dark"

from analysis import (
    anomaly_detection,
    cohort_analysis,
    eda,
    forecasting,
    profiling,
    root_cause_analysis,
    segmentation,
    significance_testing,
    whatif_simulation,
)
from core.chart_selector import choose_chart
from core.file_loader import CSVParseError, read_csv_robust
from core.intent_parser import parse_intent
from core.join_planner import compute_pairwise_overlap_matrix, execute_joins, infer_join_candidates, plan_joins
from core.llm_client import LLMClient
from core.memory import ConversationMemory, ConversationTurn
from core.self_correcting import plan_and_execute_with_retries
from core.trust_score import compute_trust_score
from core.validation import sanitize_column_name, validate_join_plan

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="🧭")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --accent: #8B5CF6;
    --accent-2: #22D3EE;
    --bg-card: #141821;
    --border-glow: rgba(139, 92, 246, 0.45);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0E14 0%, #10131C 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: rgba(139, 92, 246, 0.06);
    border: 1px dashed var(--accent);
}

/* Headings: neon gradient underline */
h1 { font-weight: 800; color: #F3F4F6; letter-spacing: -0.02em; }
h2, h3 {
    font-weight: 700;
    color: #F3F4F6;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, var(--accent), var(--accent-2)) 1;
    padding-bottom: 6px;
    display: inline-block;
}

/* Horizontal radios (top navbar + in-page choices) styled as glowing pill tabs */
div[role="radiogroup"] {
    gap: 6px;
    flex-wrap: wrap;
}
div[role="radiogroup"] label {
    background-color: var(--bg-card);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 999px;
    padding: 6px 16px !important;
    margin: 2px !important;
    transition: all 0.15s ease;
}
div[role="radiogroup"] label:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--border-glow);
}
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(139,92,246,0.25), rgba(34,211,238,0.15));
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--border-glow);
}

/* Metric cards: dark with neon glow border */
[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 0 14px rgba(139, 92, 246, 0.08);
}

/* Bordered containers (dashboard tiles etc.) get the same glow-card look */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border-color: rgba(139, 92, 246, 0.25) !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.08);
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    font-weight: 600;
    border: 1px solid rgba(139, 92, 246, 0.4);
}
.stButton > button:hover {
    border-color: var(--accent);
    box-shadow: 0 0 14px var(--border-glow);
    color: #F3F4F6;
}

/* DataFrames / tables: rounded corners */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "dashboard_tiles" not in st.session_state:
    st.session_state.dashboard_tiles = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "tables" not in st.session_state:
    st.session_state.tables = {}
if "dataset_versions" not in st.session_state:
    st.session_state.dataset_versions = []
if "last_join_result" not in st.session_state:
    st.session_state.last_join_result = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)
if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()

llm = st.session_state.llm

st.markdown(
    """
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
  <div style="font-size:2rem; filter: drop-shadow(0 0 8px rgba(139,92,246,0.6));">🧭</div>
  <div>
    <div style="font-size:1.8rem; font-weight:800; color:#F3F4F6; line-height:1.1; letter-spacing:-0.02em;">AI Data Analyst</div>
    <div style="color:#A78BFA; font-size:0.95rem;">The LLM plans and explains. Pandas/SQL/scikit-learn compute every number.</div>
  </div>
</div>
<hr style="margin-top:10px; margin-bottom:16px; border:none; height:2px; background:linear-gradient(90deg,#8B5CF6,#22D3EE,transparent); box-shadow: 0 0 8px rgba(139,92,246,0.5);">
""",
    unsafe_allow_html=True,
)

PAGES = [
    "Join Tables",
    "Profiling & Cleaning",
    "Ask a Question",
    "EDA Report",
    "Forecasting",
    "Anomalies",
    "Segmentation",
    "Cohort Analysis",
    "A/B Testing",
    "Root-Cause Analysis",
    "What-If Simulation",
    "Dashboard",
    "Export Report",
    "LLM Usage",
    "Audit Log",
]

with st.sidebar:
    st.header("Upload data")
    uploaded_files = st.file_uploader("CSV or Excel (multiple files can be joined)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if uploaded_files:
        upload_signature = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("upload_signature") != upload_signature:
            # Genuinely a new upload (not just a rerun from some other widget) —
            # re-parse and reset. Otherwise leave session_state.df/tables alone
            # so applied cleaning / join results survive every later interaction.
            st.session_state.upload_signature = upload_signature
            tables = {}
            load_errors = []
            for f in uploaded_files:
                try:
                    file_df = read_csv_robust(f) if f.name.endswith(".csv") else pd.read_excel(f)
                except CSVParseError as e:
                    load_errors.append(f"{f.name}: {e}")
                    continue
                file_df.columns = [sanitize_column_name(c) for c in file_df.columns]
                table_name = sanitize_column_name(f.name.rsplit(".", 1)[0])
                tables[table_name] = file_df
            for err in load_errors:
                st.error(err)
            st.session_state.tables = tables
            if len(tables) == 1:
                st.session_state.df = next(iter(tables.values()))
                st.session_state.dataset_versions = [{"label": "v1 (raw upload)", "df": st.session_state.df.copy()}]
                st.success(f"Loaded 1 table: {len(st.session_state.df)} rows")
            elif len(tables) > 1:
                st.session_state.df = None
                st.session_state.dataset_versions = []
                st.info(f"Loaded {len(tables)} tables — go to 'Join Tables' to combine them.")

    st.header("LLM backend")
    backend = os.environ.get("LLM_BACKEND", "groq")
    st.write(f"Active: `{backend}`")
    if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set — app will degrade to manual/templated mode.")

tables = st.session_state.tables

if not tables:
    st.info("Upload a CSV or Excel file to begin. Upload two or more related files to join them.")
    st.stop()

df = st.session_state.df
available_pages = PAGES if df is not None else ["Join Tables"]

page = st.radio("Navigate", available_pages, horizontal=True, label_visibility="collapsed")
if df is None:
    st.caption("Complete a join (or upload a single file) to unlock the remaining sections.")
st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# ---------------- Join Tables ----------------
if page == "Join Tables":
    st.subheader("Multi-file Join Planner")

    def _record_join(result, steps, source: str):
        # Persist to session_state rather than rendering directly here: this
        # block only runs on the exact rerun where the button was clicked —
        # st.button() reverts to False on every later rerun (including a
        # double-click's second event), which would otherwise make the whole
        # result vanish immediately, leaving just the button behind.
        st.session_state.df = result.merged_df
        st.session_state.dataset_versions.append(
            {"label": f"v{len(st.session_state.dataset_versions) + 1} (joined)", "df": result.merged_df.copy()}
        )
        st.session_state.last_join_result = {"result": result, "steps": steps, "source": source}

    if len(tables) < 2:
        st.info("Only one file is loaded — it's used directly. Upload a second related file to plan a join.")
    else:
        table_names = list(tables.keys())

        st.markdown("### Dimension compatibility")
        st.caption("Every column-pair's real value-overlap ratio between two tables — shows which dimensions are (and aren't) joinable, not just the ones strong enough to auto-suggest.")
        mc1, mc2 = st.columns(2)
        matrix_left = mc1.selectbox("Table A", options=table_names, key="matrix_left")
        matrix_right = mc2.selectbox("Table B", options=[t for t in table_names if t != matrix_left], key="matrix_right")
        if matrix_right:
            overlap_matrix = compute_pairwise_overlap_matrix(tables[matrix_left], tables[matrix_right])
            st.plotly_chart(
                px.imshow(overlap_matrix, text_auto=True, color_continuous_scale="Greens", labels=dict(x=matrix_right, y=matrix_left, color="overlap")),
                use_container_width=True,
            )

        st.markdown("### Option A: Let the AI plan the join")
        candidates = infer_join_candidates(tables)
        st.caption("Candidate keys are found by real value overlap between columns — the LLM only picks among these, it never invents a key.")
        if not candidates:
            st.warning("No overlapping keys were found between the uploaded tables — nothing to auto-join. Try the manual option below.")
        else:
            st.dataframe(pd.DataFrame([vars(c) for c in candidates]), use_container_width=True)
            if st.button("Plan & execute join with AI", key="ai_join_btn"):
                steps = plan_joins(tables, candidates, llm)
                result = execute_joins(tables, steps)
                if result.error:
                    st.error(result.error)
                else:
                    _record_join(result, steps, source="ai")

        st.markdown("### Option B: Choose the join yourself")
        st.caption("Pick the tables, keys, and join type directly — bypasses the AI planner entirely.")
        jc1, jc2 = st.columns(2)
        with jc1:
            manual_left_table = st.selectbox("Left table", options=table_names, key="manual_left_table")
            manual_left_key = st.selectbox("Left key", options=tables[manual_left_table].columns.tolist(), key="manual_left_key")
        with jc2:
            manual_right_table = st.selectbox("Right table", options=[t for t in table_names if t != manual_left_table], key="manual_right_table")
            manual_right_key = st.selectbox("Right key", options=tables[manual_right_table].columns.tolist(), key="manual_right_key")
        manual_how = st.radio(
            "Join type",
            options=["inner", "left", "right", "outer"],
            format_func=lambda h: {"inner": "Inner (only matching rows)", "left": "Left (all of left table)", "right": "Right (all of right table)", "outer": "Outer (all rows, matched or not)"}[h],
            horizontal=True,
            key="manual_how",
        )
        if st.button("Execute this join", key="manual_join_btn"):
            manual_plan = {
                "joins": [
                    {
                        "left_table": manual_left_table,
                        "right_table": manual_right_table,
                        "left_key": manual_left_key,
                        "right_key": manual_right_key,
                        "how": manual_how,
                    }
                ]
            }
            try:
                steps = validate_join_plan(manual_plan, tables)
                result = execute_joins(tables, steps)
                if result.error:
                    st.error(result.error)
                else:
                    _record_join(result, steps, source="manual")
            except Exception as e:
                st.error(str(e))

        last_join = st.session_state.get("last_join_result")
        if last_join:
            result, steps, source = last_join["result"], last_join["steps"], last_join["source"]
            if len(result.merged_df) == 0:
                st.warning(
                    "This join produced 0 rows — the key columns matched no values in common. "
                    "Check for mismatched types (e.g. numbers vs text), extra whitespace, or different casing "
                    "between the two key columns, or try a different key pair / join type."
                )
            else:
                st.success(f"Joined into {len(result.merged_df)} rows x {len(result.merged_df.columns)} columns. Use 'Navigate' in the sidebar to continue.")

            st.markdown("**Joined result preview**")
            st.dataframe(result.merged_df.head(50), use_container_width=True, key=f"join_preview_{source}")

            row_counts = pd.DataFrame(
                {
                    "table": list(tables.keys()) + ["JOINED RESULT"],
                    "rows": [len(t) for t in tables.values()] + [len(result.merged_df)],
                }
            )
            st.plotly_chart(px.bar(row_counts, x="table", y="rows", color="table"), use_container_width=True, key=f"join_row_count_chart_{source}")

            with st.expander("Join plan & code (trust trace)"):
                st.json([vars(s) for s in steps])
                st.code(result.code_executed)

# ---------------- Profiling & Cleaning ----------------
elif page == "Profiling & Cleaning":
    st.subheader("Data Profile")
    report = profiling.profile_dataframe(df)

    if "quality_history" not in st.session_state:
        st.session_state.quality_history = []
    if not st.session_state.quality_history or st.session_state.quality_history[-1]["quality_score"] != report.quality_score:
        st.session_state.quality_history.append(
            {"version": len(st.session_state.quality_history) + 1, "quality_score": report.quality_score}
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", report.row_count)
    c2.metric("Columns", report.column_count)
    c3.metric("Duplicate rows", report.duplicate_rows)

    st.subheader("Data quality score")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Overall", f"{report.quality_score}/100")
    q2.metric("Completeness", f"{report.completeness_score}/100")
    q3.metric("Consistency", f"{report.consistency_score}/100")
    q4.metric("Validity", f"{report.validity_score}/100")
    if len(st.session_state.quality_history) > 1:
        st.line_chart(pd.DataFrame(st.session_state.quality_history).set_index("version")["quality_score"])
        st.caption("Quality score across cleaning versions applied this session.")

    st.dataframe(pd.DataFrame([vars(c) for c in report.columns]), use_container_width=True)

    if report.suggestions:
        st.subheader("Suggested cleaning actions")
        for s in report.suggestions:
            st.write(f"- {s}")

        with st.form("cleaning_form"):
            drop_dupes = st.checkbox("Drop duplicate rows", value=report.duplicate_rows > 0)
            drop_cols = st.multiselect("Drop columns", options=list(df.columns))
            pii_cols = [c.name for c in report.columns if c.likely_pii]
            mask_cols = st.multiselect("Mask likely-PII columns", options=pii_cols, default=pii_cols)
            missing_cols = [c.name for c in report.columns if c.missing_count > 0]
            impute_actions = {}
            if missing_cols:
                st.write("Missing-value handling:")
                for col_name in missing_cols:
                    impute_actions[col_name] = st.selectbox(
                        f"'{col_name}'", options=["none", "mean", "median", "mode", "drop_rows"], key=f"impute_{col_name}"
                    )
            apply_btn = st.form_submit_button("Apply cleaning")
        if apply_btn:
            impute = {c: m for c, m in impute_actions.items() if m != "none"}
            actions = {"drop_duplicates": drop_dupes, "drop_columns": drop_cols, "mask_pii": mask_cols, "impute": impute}
            st.session_state.df = profiling.apply_cleaning(df, actions)
            st.session_state.dataset_versions.append(
                {"label": f"v{len(st.session_state.dataset_versions) + 1} (cleaned)", "df": st.session_state.df.copy()}
            )
            st.success("Cleaning applied.")
            st.rerun()

    if len(st.session_state.dataset_versions) > 1:
        st.subheader("Version history")
        st.caption("Every join and cleaning action is snapshotted this session — roll back to any earlier version.")
        for i, version in enumerate(st.session_state.dataset_versions):
            vc1, vc2, vc3 = st.columns([2, 2, 1])
            vc1.write(version["label"])
            vc2.write(f"{len(version['df'])} rows x {len(version['df'].columns)} cols")
            is_current = i == len(st.session_state.dataset_versions) - 1
            if not is_current and vc3.button("Roll back", key=f"rollback_{i}"):
                st.session_state.df = version["df"].copy()
                st.session_state.dataset_versions.append({"label": f"v{len(st.session_state.dataset_versions) + 1} (rolled back to {version['label']})", "df": version["df"].copy()})
                st.rerun()

# ---------------- Ask a Question ----------------
elif page == "Ask a Question":
    st.subheader("Ask a question about your data")
    st.caption("Remembers the last 5 turns, so follow-ups like 'now break that down by region' work without repeating context.")
    if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        st.info(
            "No LLM configured, so questions can't be interpreted — every question below will fall back to a generic "
            "data summary instead of a targeted answer. Set GROQ_API_KEY (see .env.example) for real Q&A, or use the "
            "other sidebar sections (EDA Report, Forecasting, Segmentation, etc.) which don't need an LLM at all."
        )
    question = st.text_input("e.g. 'What is the average revenue by region?'")
    col_ask, col_clear = st.columns([1, 1])
    ask_clicked = col_ask.button("Ask")
    if col_clear.button("Clear conversation memory"):
        st.session_state.memory.clear()
        st.success("Conversation memory cleared.")

    if ask_clicked and question:
        memory = st.session_state.memory
        intent = parse_intent(question, llm, memory=memory)
        outcome = plan_and_execute_with_retries(intent, df, llm, memory=memory)
        plan, result = outcome.plan, outcome.result

        from core.response_composer import compose_response

        answer = compose_response(question, result, llm)

        trust = compute_trust_score(outcome)

        memory.add(ConversationTurn(question=question, intent_type=intent.intent_type, op_type=plan.op_type, answer=answer))
        st.session_state.chat_history.append({"question": question, "intent": intent.intent_type, "answer": answer})
        st.session_state.audit_log.append(
            {
                "question": question,
                "intent": intent.intent_type,
                "op_type": plan.op_type,
                "code": result.code_executed,
                "error": result.error,
                "attempts": outcome.attempts,
                "trust_score": trust.score,
            }
        )
        st.session_state.last_qa_result = {
            "question": question,
            "answer": answer,
            "result_df": result.result_df,
            "error": result.error,
            "attempts": outcome.attempts,
            "attempt_log": outcome.attempt_log,
            "plan": vars(plan),
            "code": result.code_executed,
            "trust": trust,
        }

    last = st.session_state.get("last_qa_result")
    if last:
        st.markdown(f"**Answer:** {last['answer']}")
        if last.get("trust"):
            trust = last["trust"]
            badge = {"High": "🟢", "High (self-corrected)": "🟢", "Medium": "🟡", "Low": "🟠", "Failed": "🔴"}.get(trust.label, "⚪")
            st.caption(f"{badge} Trust: **{trust.label}** ({trust.score}/100) — {trust.reason}")
        if last.get("attempts", 1) > 1:
            st.caption(f"Self-corrected after {last['attempts']} attempts — see the trust trace below for what was tried.")
        if last["error"]:
            st.error(last["error"])
        elif not last["result_df"].empty:
            st.dataframe(last["result_df"], use_container_width=True)
            spec = choose_chart(last["result_df"])
            fig = None
            if spec.chart_type == "bar":
                fig = px.bar(last["result_df"], x=spec.x, y=spec.y, color=spec.color)
            elif spec.chart_type == "line":
                fig = px.line(last["result_df"], x=spec.x, y=spec.y)
            elif spec.chart_type == "scatter":
                fig = px.scatter(last["result_df"], x=spec.x, y=spec.y)
            elif spec.chart_type == "histogram":
                fig = px.histogram(last["result_df"], x=spec.x)
            elif spec.chart_type == "heatmap":
                pivot = last["result_df"].pivot_table(index=spec.y, columns=spec.x, values=spec.color, aggfunc="sum")
                fig = px.imshow(pivot, text_auto=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Chart auto-selected: {spec.chart_type} — {spec.reason}")
            if st.button("Pin this result to Dashboard"):
                st.session_state.dashboard_tiles.append(
                    {"title": last["question"], "df": last["result_df"], "chart_spec": spec, "answer": last["answer"]}
                )
                st.success("Pinned — see it under 'Dashboard' in the sidebar.")
        with st.expander("Show operation plan & code (trust trace)"):
            if last.get("attempt_log") and len(last["attempt_log"]) > 1:
                st.write("Self-correction attempts:")
                st.dataframe(pd.DataFrame(last["attempt_log"]), use_container_width=True)
            st.json(last["plan"])
            st.code(last["code"] or "(none)")

    if st.session_state.chat_history:
        st.subheader("Session history")
        for turn in reversed(st.session_state.chat_history[-10:]):
            st.write(f"**Q ({turn['intent']}):** {turn['question']}")
            st.write(f"**A:** {turn['answer']}")
            st.divider()

# ---------------- EDA Report ----------------
elif page == "EDA Report":
    st.subheader("Automated EDA")
    eda_report = eda.run_eda(df)
    for note in eda_report.narrative:
        st.write(f"- {note}")
    if not eda_report.numeric_summary.empty:
        st.write("Numeric summary")
        st.dataframe(eda_report.numeric_summary, use_container_width=True)
    if not eda_report.correlation_matrix.empty:
        st.write("Correlation matrix")
        st.plotly_chart(px.imshow(eda_report.correlation_matrix, text_auto=True), use_container_width=True)
    st.write("Missingness")
    st.dataframe(eda_report.missingness, use_container_width=True)

# ---------------- Forecasting ----------------
elif page == "Forecasting":
    st.subheader("Explainable Forecasting")
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_col = st.selectbox("Date column", options=date_cols or df.columns.tolist())
    value_col = st.selectbox("Value column", options=numeric_cols)
    periods = st.slider("Periods to forecast", 3, 24, 12)
    if st.button("Run forecast"):
        try:
            fc = forecasting.run_forecast(df, date_col, value_col, periods=periods)
            hist = fc.history.assign(kind="history").rename(columns={"y": "value"})
            fut = fc.forecast.assign(kind="forecast").rename(columns={"yhat": "value"})
            combined = pd.concat([hist[["ds", "value", "kind"]], fut[["ds", "value", "kind"]]])
            fig = px.line(combined, x="ds", y="value", color="kind")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Trend is **{fc.trend_direction}**. Seasonality detected: {fc.seasonality_detected}.")
            st.caption(fc.confidence_note)
            st.dataframe(fc.forecast, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# ---------------- Anomalies ----------------
elif page == "Anomalies":
    st.subheader("Anomaly Detection")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    method = st.radio("Method", ["Z-score", "IQR", "Isolation Forest"], horizontal=True)
    if method == "Isolation Forest":
        cols = st.multiselect("Columns", options=numeric_cols, default=numeric_cols[:2])
        if st.button("Detect anomalies") and cols:
            result = anomaly_detection.detect_isolation_forest(df, cols)
            st.write(f"Flagged {result.total_flagged} rows ({result.flagged_pct}%) using {result.method}.")
            st.dataframe(result.flagged_rows, use_container_width=True)
    else:
        col = st.selectbox("Column", options=numeric_cols)
        if st.button("Detect anomalies"):
            fn = anomaly_detection.detect_zscore if method == "Z-score" else anomaly_detection.detect_iqr
            result = fn(df, col)
            st.write(f"Flagged {result.total_flagged} rows ({result.flagged_pct}%) using {result.method}.")
            st.dataframe(result.flagged_rows, use_container_width=True)

# ---------------- Segmentation ----------------
elif page == "Segmentation":
    st.subheader("RFM & K-Means Segmentation")
    cols = df.columns.tolist()
    numeric_cols_seg = df.select_dtypes(include="number").columns.tolist()
    date_like = [c for c in cols if "date" in c.lower() or "time" in c.lower()]
    default_date_idx = cols.index(date_like[0]) if date_like else min(1, len(cols) - 1)

    customer_col = st.selectbox("Customer ID column", options=cols)
    date_col2 = st.selectbox("Transaction date column", options=cols, index=default_date_idx, key="seg_date")
    amount_col = st.selectbox("Amount column", options=numeric_cols_seg)
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    if st.button("Run segmentation"):
        try:
            rfm = segmentation.compute_rfm(df, customer_col, date_col2, amount_col)
            seg = segmentation.kmeans_segment(rfm, n_clusters=n_clusters)
            st.write("Cluster profiles")
            st.dataframe(seg.cluster_profiles, use_container_width=True)
            fig = px.scatter(seg.labeled_table, x="recency", y="monetary", color="label", size="frequency")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(seg.labeled_table, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# ---------------- Cohort Analysis ----------------
elif page == "Cohort Analysis":
    st.subheader("Retention Cohorts")
    st.caption("Customers are grouped by the period of their first activity; retention is the real fraction of each cohort active in later periods.")
    cols = df.columns.tolist()
    cohort_customer_col = st.selectbox("Customer ID column", options=cols, key="cohort_customer_col")
    cohort_date_col = st.selectbox("Activity date column", options=cols, key="cohort_date_col")
    cohort_period = st.selectbox("Cohort period", options=["M", "W", "D"], format_func=lambda p: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[p])
    if st.button("Build retention cohorts"):
        try:
            result = cohort_analysis.build_retention_cohorts(df, cohort_customer_col, cohort_date_col, period=cohort_period)
            st.write("Cohort sizes")
            st.dataframe(result.cohort_sizes.to_frame("size"), use_container_width=True)
            st.write("Retention % by periods since first activity")
            fig = px.imshow(result.retention_pct, text_auto=True, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result.retention_counts, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# ---------------- A/B Testing ----------------
elif page == "A/B Testing":
    st.subheader("A/B Testing / Statistical Significance")
    st.caption("Real t-tests and chi-square tests via scipy — the significance verdict is computed, never guessed.")
    test_kind = st.radio("Test type", ["Compare two groups (t-test)", "Compare two time periods (t-test)", "Association between two categories (chi-square)"])

    if test_kind == "Compare two groups (t-test)":
        cols = df.columns.tolist()
        group_col = st.selectbox("Group column", options=cols, key="ab_group_col")
        value_col = st.selectbox("Metric column", options=df.select_dtypes(include="number").columns.tolist(), key="ab_value_col")
        options = df[group_col].dropna().unique().tolist() if group_col else []
        group_a = st.selectbox("Group A", options=options, key="ab_group_a")
        group_b = st.selectbox("Group B", options=[o for o in options if o != group_a], key="ab_group_b")
        if st.button("Run t-test"):
            try:
                result = significance_testing.compare_means(df, group_col, value_col, group_a, group_b)
                st.write(result.summary)
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{result.group_a_label} mean (n={result.group_a_n})", result.group_a_mean)
                c2.metric(f"{result.group_b_label} mean (n={result.group_b_n})", result.group_b_mean)
                c3.metric("p-value", result.p_value)
                st.caption(f"t-statistic={result.t_statistic}, Cohen's d={result.cohens_d}")
            except ValueError as e:
                st.error(str(e))

    elif test_kind == "Compare two time periods (t-test)":
        cols = df.columns.tolist()
        date_col3 = st.selectbox("Date column", options=cols, key="ab_date_col")
        value_col2 = st.selectbox("Metric column", options=df.select_dtypes(include="number").columns.tolist(), key="ab_value_col2")
        c1, c2 = st.columns(2)
        with c1:
            period_a_start = st.date_input("Period A start", key="pa_start")
            period_a_end = st.date_input("Period A end", key="pa_end")
        with c2:
            period_b_start = st.date_input("Period B start", key="pb_start")
            period_b_end = st.date_input("Period B end", key="pb_end")
        if st.button("Run t-test on periods"):
            try:
                result = significance_testing.compare_time_periods(
                    df, date_col3, value_col2, period_a_start, period_a_end, period_b_start, period_b_end
                )
                st.write(result.summary)
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Period A mean (n={result.group_a_n})", result.group_a_mean)
                c2.metric(f"Period B mean (n={result.group_b_n})", result.group_b_mean)
                c3.metric("p-value", result.p_value)
            except ValueError as e:
                st.error(str(e))

    else:
        cols = df.columns.tolist()
        col_a = st.selectbox("Category A", options=cols, key="chi_col_a")
        col_b = st.selectbox("Category B", options=[c for c in cols if c != col_a], key="chi_col_b")
        if st.button("Run chi-square test"):
            result = significance_testing.chi_square_test(df, col_a, col_b)
            st.write(result.summary)
            st.write(f"chi2={result.chi2_statistic}, dof={result.degrees_of_freedom}, p-value={result.p_value}")
            st.dataframe(result.contingency_table, use_container_width=True)

# ---------------- Root-Cause Analysis ----------------
elif page == "Root-Cause Analysis":
    st.subheader("Root-Cause / Driver Analysis")
    st.caption("Decomposes a metric's change between two periods by a dimension, computed directly from the real per-period sums — not estimated.")
    cols = df.columns.tolist()
    rc_date_col = st.selectbox("Date column", options=cols, key="rc_date_col")
    rc_metric_col = st.selectbox("Metric column", options=df.select_dtypes(include="number").columns.tolist(), key="rc_metric_col")
    rc_dim_col = st.selectbox("Dimension to decompose by", options=[c for c in cols if c not in (rc_date_col, rc_metric_col)], key="rc_dim_col")
    c1, c2 = st.columns(2)
    with c1:
        rc_a_start = st.date_input("Period A start", key="rc_a_start")
        rc_a_end = st.date_input("Period A end", key="rc_a_end")
    with c2:
        rc_b_start = st.date_input("Period B start", key="rc_b_start")
        rc_b_end = st.date_input("Period B end", key="rc_b_end")
    if st.button("Decompose change"):
        try:
            result = root_cause_analysis.decompose_change(
                df, rc_date_col, rc_metric_col, rc_dim_col, rc_a_start, rc_a_end, rc_b_start, rc_b_end
            )
            st.write(result.summary)
            drivers_df = pd.DataFrame([vars(d) for d in result.drivers])
            fig = px.bar(drivers_df, x="dimension_value", y="change", color="change", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(drivers_df, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# ---------------- What-If Simulation ----------------
elif page == "What-If Simulation":
    st.subheader("What-If / Scenario Simulation")
    st.caption("Projects the impact of changing one input using a fitted linear regression model. Always a labeled simulation, never a measured result.")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.info("What-if simulation needs at least two numeric columns (one input, one target). This dataset only has one.")
    else:
        wi_input_col = st.selectbox("Input variable to change", options=numeric_cols, key="wi_input_col")
        wi_target_col = st.selectbox("Target metric", options=[c for c in numeric_cols if c != wi_input_col], key="wi_target_col")
        wi_pct = st.slider("Change input by (%)", -50, 100, 10, key="wi_pct")
        if st.button("Run simulation"):
            try:
                result = whatif_simulation.simulate_variable_change(df, wi_input_col, wi_target_col, wi_pct)
                st.warning(result.summary)
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline (projected)", round(result.baseline_prediction, 2))
                c2.metric("Scenario (projected)", round(result.scenario_prediction, 2), delta=f"{result.pct_change:+.1f}%")
                c3.metric("Model R²", result.r_squared)
            except ValueError as e:
                st.error(str(e))

# ---------------- Dashboard ----------------
elif page == "Dashboard":
    st.subheader("Dashboard")

    PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#06B6D4", "#8B5CF6", "#EF4444"]
    CHART_HEIGHT = 230

    def _compact(fig):
        fig.update_layout(height=CHART_HEIGHT, margin=dict(l=10, r=10, t=34, b=10), legend=dict(orientation="h", y=-0.25))
        return fig

    if len(tables) >= 2:
        st.markdown("### Dataset Comparison Overview")
        st.caption(f"Auto-generated comparison across all {len(tables)} uploaded files — computed directly from each table, no LLM involved.")

        table_names = list(tables.keys())
        profiles = {name: profiling.profile_dataframe(t) for name, t in tables.items()}

        overview_df = pd.DataFrame(
            {
                "table": table_names,
                "rows": [len(tables[n]) for n in table_names],
                "columns": [len(tables[n].columns) for n in table_names],
                "numeric_columns": [len(tables[n].select_dtypes(include="number").columns) for n in table_names],
                "categorical_columns": [
                    len(tables[n].columns) - len(tables[n].select_dtypes(include="number").columns) for n in table_names
                ],
                "missing_pct": [
                    round(tables[n].isna().sum().sum() / (len(tables[n]) * len(tables[n].columns)) * 100, 2)
                    if len(tables[n]) and len(tables[n].columns)
                    else 0.0
                    for n in table_names
                ],
                "duplicate_rows": [profiles[n].duplicate_rows for n in table_names],
                "quality_score": [profiles[n].quality_score for n in table_names],
                "completeness_score": [profiles[n].completeness_score for n in table_names],
                "consistency_score": [profiles[n].consistency_score for n in table_names],
                "validity_score": [profiles[n].validity_score for n in table_names],
            }
        )

        row1 = st.columns(3)
        row2 = st.columns(2)

        # 1. Grouped bar — rows & columns per table, sorted, with value labels + a share-of-total in hover
        size_long = overview_df.melt(id_vars="table", value_vars=["rows", "columns"], var_name="metric", value_name="value")
        size_long = size_long.sort_values(["metric", "value"], ascending=[True, False])
        fig1 = px.bar(
            size_long, x="table", y="value", color="metric", barmode="group", title="Size", text="value",
            color_discrete_sequence=PALETTE,
        )
        fig1.update_traces(textposition="outside")
        row1[0].plotly_chart(_compact(fig1), use_container_width=True)

        # 2. Donut — share of total rows per table, with count + percent both visible
        fig2 = px.pie(
            overview_df, names="table", values="rows", hole=0.55, title="Row share", color="table",
            color_discrete_sequence=PALETTE, hover_data=["rows"],
        )
        fig2.update_traces(textinfo="label+percent", texttemplate="%{label}<br>%{percent}")
        row1[1].plotly_chart(_compact(fig2), use_container_width=True)

        # 3. Heatmap — column-type composition (table x numeric/categorical), annotated + labeled axes
        dtype_matrix = overview_df.set_index("table")[["numeric_columns", "categorical_columns"]]
        dtype_matrix.columns = ["numeric", "categorical"]
        fig3 = px.imshow(
            dtype_matrix, text_auto=True, title="Column types", color_continuous_scale=["#EEF2FF", PALETTE[0]],
            labels=dict(x="column type", y="table", color="count"),
        )
        row1[2].plotly_chart(_compact(fig3), use_container_width=True)

        # 4. Radar/polar — quality profile per table, with exact scores on hover
        radar_categories = ["Completeness", "Consistency", "Validity", "Overall"]
        radar_fig = go.Figure()
        for i, name in enumerate(table_names):
            row = overview_df[overview_df["table"] == name].iloc[0]
            values = [row["completeness_score"], row["consistency_score"], row["validity_score"], row["quality_score"]]
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=radar_categories + [radar_categories[0]],
                    fill="toself",
                    name=name,
                    line_color=PALETTE[i % len(PALETTE)],
                    hovertemplate="%{theta}: %{r:.1f}/100<extra>" + name + "</extra>",
                )
            )
        radar_fig.update_layout(title="Quality profile", polar=dict(radialaxis=dict(range=[0, 100])))
        row2[0].plotly_chart(_compact(radar_fig), use_container_width=True)

        # 5. Scatter — rows vs missing%, sized by duplicate rows, labeled with table name directly on the point
        fig5 = px.scatter(
            overview_df, x="rows", y="missing_pct", size="duplicate_rows", color="table", text="table",
            title="Rows vs missing % (size = duplicates)", color_discrete_sequence=PALETTE, size_max=28,
            hover_data=["quality_score", "duplicate_rows"],
        )
        fig5.update_traces(textposition="top center")
        row2[1].plotly_chart(_compact(fig5), use_container_width=True)

        with st.expander("Underlying numbers"):
            st.dataframe(overview_df, use_container_width=True)

        st.divider()
    elif len(tables) == 1:
        st.caption("Upload a second related file to see an automatic comparison overview here.")

    if len(st.session_state.dataset_versions) > 1:
        st.markdown("### Version History")
        st.caption("Every join, cleaning action, and rollback this session creates a new snapshot — shown here so every previous state stays visible, not just the current one.")
        versions = st.session_state.dataset_versions
        version_df = pd.DataFrame(
            {
                "version": [f"{i + 1}. {v['label']}" for i, v in enumerate(versions)],
                "rows": [len(v["df"]) for v in versions],
                "columns": [len(v["df"].columns) for v in versions],
            }
        )
        version_long = version_df.melt(id_vars="version", value_vars=["rows", "columns"], var_name="metric", value_name="value")
        vfig = px.bar(
            version_long, x="version", y="value", color="metric", barmode="group", text="value",
            title="Dataset size across every version this session", color_discrete_sequence=PALETTE,
        )
        vfig.update_traces(textposition="outside")
        vfig.update_layout(height=300, xaxis_tickangle=-25, margin=dict(l=10, r=10, t=40, b=80))
        st.plotly_chart(vfig, use_container_width=True)
        with st.expander("Version details"):
            st.dataframe(version_df, use_container_width=True)
        st.divider()

    st.markdown("### Pinned Q&A Results")
    st.caption("Pin any Q&A result (from 'Ask a Question') here as a persistent tile.")
    if not st.session_state.dashboard_tiles:
        st.info("No tiles pinned yet.")
    else:
        if st.button("Clear dashboard"):
            st.session_state.dashboard_tiles = []
            st.rerun()
        for i, tile in enumerate(st.session_state.dashboard_tiles):
            with st.container(border=True):
                st.markdown(f"**{tile['title']}**")
                st.caption(tile["answer"])
                spec = tile["chart_spec"]
                tile_df = tile["df"]
                fig = None
                if spec.chart_type == "bar":
                    fig = px.bar(tile_df, x=spec.x, y=spec.y, color=spec.color)
                elif spec.chart_type == "line":
                    fig = px.line(tile_df, x=spec.x, y=spec.y)
                elif spec.chart_type == "scatter":
                    fig = px.scatter(tile_df, x=spec.x, y=spec.y)
                elif spec.chart_type == "histogram":
                    fig = px.histogram(tile_df, x=spec.x)
                elif spec.chart_type == "heatmap":
                    pivot = tile_df.pivot_table(index=spec.y, columns=spec.x, values=spec.color, aggfunc="sum")
                    fig = px.imshow(pivot, text_auto=True)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_tile_{i}")
                else:
                    st.dataframe(tile_df, use_container_width=True)
                if st.button("Remove", key=f"remove_tile_{i}"):
                    st.session_state.dashboard_tiles.pop(i)
                    st.rerun()

# ---------------- Export Report ----------------
elif page == "Export Report":
    st.subheader("Export Report")
    st.caption("Bundles the EDA narrative, data profile, and this session's Q&A history into a self-contained HTML file.")
    if st.button("Generate report"):
        from reports.export import build_html_report

        eda_report_for_export = eda.run_eda(df)
        profile_for_export = profiling.profile_dataframe(df)
        html = build_html_report(
            eda_narrative=eda_report_for_export.narrative,
            chat_history=st.session_state.chat_history,
            profile_summary={
                "row_count": profile_for_export.row_count,
                "column_count": profile_for_export.column_count,
                "duplicate_rows": profile_for_export.duplicate_rows,
            },
        )
        st.session_state.generated_report_html = html
        st.success("Report generated below — copy or save the HTML.")
    if st.session_state.get("generated_report_html"):
        st.download_button(
            "Download report as HTML",
            data=st.session_state.generated_report_html,
            file_name="ai_data_analyst_report.html",
            mime="text/html",
        )
        with st.expander("Preview HTML source"):
            st.code(st.session_state.generated_report_html, language="html")

# ---------------- LLM Usage ----------------
elif page == "LLM Usage":
    st.subheader("LLM Cost & Token Usage")
    st.caption("Every LLM call this session — success or failure — logged with backend, latency, and token estimate, so the AI part of the app isn't a hidden line item.")
    usage_log = llm.usage_log
    if not usage_log:
        st.info("No LLM calls made yet this session.")
    else:
        usage_df = pd.DataFrame(usage_log)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calls", len(usage_df))
        c2.metric("Total tokens (est.)", int(usage_df["tokens_estimate"].sum()))
        c3.metric("Avg latency (ms)", round(usage_df["latency_ms"].mean(), 1))
        c4.metric("Failed calls", int((~usage_df["success"]).sum()))
        st.line_chart(usage_df["tokens_estimate"].cumsum())
        st.caption("Cumulative estimated tokens across this session's LLM calls.")
        st.dataframe(usage_df, use_container_width=True)

# ---------------- Audit Log ----------------
elif page == "Audit Log":
    st.subheader("Query Audit Log")
    st.caption("Every question, its operation plan, and the executed code — for full traceability.")
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
    else:
        st.write("No queries yet.")
