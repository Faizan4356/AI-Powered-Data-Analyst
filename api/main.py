"""FastAPI layer exposing the same Q&A pipeline outside the Streamlit UI.

Stateless per-request: caller uploads/references a dataset, the same
intent_parser -> operation_planner -> executor -> response_composer
pipeline used by app.py runs, and only the composed answer + executed
code (audit trace) is returned.
"""
import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.file_loader import CSVParseError, read_csv_robust
from core.intent_parser import parse_intent
from core.llm_client import LLMClient
from core.memory import ConversationMemory, ConversationTurn
from core.response_composer import compose_response
from core.self_correcting import plan_and_execute_with_retries
from core.validation import sanitize_column_name

app = FastAPI(title="AI Data Analyst API")
llm = LLMClient()

# In-memory store keyed by dataset_id. A real deployment would use object
# storage / a DB; this keeps the API layer self-contained for the demo.
_DATASETS: dict[str, pd.DataFrame] = {}
# One rolling conversation history per (dataset_id, session_id) pair.
_MEMORIES: dict[str, ConversationMemory] = {}


class AskRequest(BaseModel):
    dataset_id: str
    question: str
    session_id: str = "default"


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.endswith(".csv"):
        try:
            df = read_csv_robust(io.BytesIO(content))
        except CSVParseError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        df = pd.read_excel(io.BytesIO(content))
    df.columns = [sanitize_column_name(c) for c in df.columns]
    dataset_id = file.filename
    _DATASETS[dataset_id] = df
    return {"dataset_id": dataset_id, "rows": len(df), "columns": list(df.columns)}


@app.post("/ask")
def ask(req: AskRequest):
    df = _DATASETS.get(req.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id — upload it first via /datasets")

    memory_key = f"{req.dataset_id}::{req.session_id}"
    memory = _MEMORIES.setdefault(memory_key, ConversationMemory(max_turns=5))

    intent = parse_intent(req.question, llm, memory=memory)
    outcome = plan_and_execute_with_retries(intent, df, llm, memory=memory)
    plan, result = outcome.plan, outcome.result
    answer = compose_response(req.question, result, llm)

    memory.add(ConversationTurn(question=req.question, intent_type=intent.intent_type, op_type=plan.op_type, answer=answer))

    return {
        "answer": answer,
        "intent": intent.intent_type,
        "op_type": plan.op_type,
        "code_executed": result.code_executed,
        "result": result.result_df.to_dict(orient="records") if not result.result_df.empty else [],
        "error": result.error,
        "attempts": outcome.attempts,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
