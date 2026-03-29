# src/main_graph.py
# Vanilla RAG version
# 流程：query -> retrieve -> generate -> finalize

import os
from typing import TypedDict, Optional, Dict, Any, List
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END

from tools import retrieve_process_tool, generate_answer_tool
from result_logger import append_graph_result_to_excel

# ---------------------------
# 基本設定
# ---------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
BATCH_QUESTION_PATH = os.path.join(PROJECT_ROOT, "data", "題目.xlsx")
RESULT_EXCEL_PATH = "rag_results_vanilla.xlsx"

# ---------------------------
# Graph State
# ---------------------------
class GraphState(TypedDict, total=False):
    query_original: str
    product_id: Optional[str]

    retrieved: Dict[str, Any]
    graded: Dict[str, Any]
    expanded_docs: List[Dict[str, Any]]
    context: str

    answer: str
    logs: List[str]

    # 批次模式可選欄位
    row_index: int

# ---------------------------
# 小工具：log
# ---------------------------
def _log(state: GraphState, msg: str) -> GraphState:
    logs = state.get("logs", [])
    logs.append(msg)
    state["logs"] = logs
    print(msg)
    return state

# ---------------------------
# 小工具：找欄位名稱
# ---------------------------
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in df.columns:
        if str(col).strip() in candidates:
            return col
    return None

# ---------------------------
# Nodes
# ---------------------------
def n_init(state: GraphState) -> GraphState:
    state["start_time"] = time.time()
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    state["logs"] = []
    state["retrieved"] = {}
    state["graded"] = {}

    pid = state.get("product_id")
    row_index = state.get("row_index")

    if row_index is not None:
        return _log(state, f"📌 [第 {row_index} 題] 已鎖定產品名稱：{pid}")
    return _log(state, f"📌 已鎖定產品名稱：{pid}")

def n_retrieve_process(state: GraphState) -> GraphState:
    q = state["query_original"]

    out = retrieve_process_tool(
        query=q,
        product_id=state.get("product_id"),
        threshold=0.3,
        k_retrieve=5,
        max_expand=10,
    )

    state["graded"] = out
    state["context"] = out.get("context", "")
    state["retrieved"] = {"docs": out.get("retrieved_docs", [])}
    state["expanded_docs"] = out.get("expanded_docs", [])

    print("\n========== [Retrieved Context Preview] ==========")
    preview = state["context"][:1000] if state.get("context") else ""
    print(preview if preview else "(empty context)")

    return state

def n_generate(state: GraphState) -> GraphState:
    out = generate_answer_tool(
        query=state["query_original"],
        context=state.get("context", ""),
    )

    state["answer"] = out.get("answer", "")

    print("\n========== [Generated Answer] ==========")
    print(state["answer"])

    return state

def n_finalize(state: GraphState) -> GraphState:
    end_time = time.time()
    start_time = state.get("start_time")
    if start_time:
        state["latency"] = round(end_time - start_time, 2)
    else:
        state["latency"] = None

    print("\n================= Final Response =================")
    print("【答案】")
    print(state.get("answer", ""))

    print("\n【來源（Top context chunks）】")
    expanded = state.get("expanded_docs", [])
    if isinstance(expanded, list) and expanded:
        for i, d in enumerate(expanded[:5]):
            cid = (d.get("meta", {}) or {}).get("CHUNK_ID")
            title = (d.get("meta", {}) or {}).get("TITLE", "")
            print(f"- [{i}] CHUNK_ID={cid} {title}".strip())
    else:
        print("(無)")
    print("=====================================================\n")

    try:
        excel_path = append_graph_result_to_excel(
            state,
            excel_path="rag_results_vanilla.xlsx"
        )
        print(f"📘 已寫入 Excel：{excel_path}")
    except Exception as e:
        print(f"⚠️ Excel 寫入失敗：{e}")

    return state

# ---------------------------
# Build Graph
# ---------------------------
def build_app():
    g = StateGraph(GraphState)

    g.add_node("init", n_init)
    g.add_node("retrieve_process", n_retrieve_process)
    g.add_node("generate", n_generate)
    g.add_node("finalize", n_finalize)

    g.add_edge(START, "init")
    g.add_edge("init", "retrieve_process")
    g.add_edge("retrieve_process", "generate")
    g.add_edge("generate", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

# ---------------------------
# 單題模式
# ---------------------------
def run_single_mode(app):
    while True:
        product_id = input("🏷️ 請先輸入產品代號/名稱（Enter 離開）：").strip()
        if not product_id:
            print("👋 結束保險問答")
            break

        q = input("📝 請輸入保險問題（Enter 離開）：").strip()
        if not q:
            print("👋 結束保險問答")
            break

        state: GraphState = {
            "product_id": product_id,
            "query_original": q,
        }

        print("🔍 正在檢索並回答問題...")
        app.invoke(state)

# ---------------------------
# 批次模式
# ---------------------------
def run_batch_mode(app, excel_path: str = BATCH_QUESTION_PATH):
    if not os.path.exists(excel_path):
        print(f"❌ 找不到題目檔案：{excel_path}")
        return

    df = pd.read_excel(excel_path)
    print(f"📂 已讀取題目檔：{excel_path}")
    print(f"📊 總筆數：{len(df)}")

    # 可接受的欄位名稱
    question_col = find_column(df, ["問題", "query", "question", "題目"])
    product_col = find_column(df, ["產品代號", "product_id", "product", "商品代號", "商品名稱", "產品名稱"])

    if question_col is None:
        print("❌ 題目.xlsx 找不到問題欄位，請至少有以下其中一個欄名：")
        print("   問題 / query / question / 題目")
        return

    if product_col is None:
        print("❌ 題目.xlsx 找不到產品欄位，請至少有以下其中一個欄名：")
        print("   產品代號 / product_id / product / 商品代號 / 商品名稱 / 產品名稱")
        return

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        q = str(row[question_col]).strip() if pd.notna(row[question_col]) else ""
        product_id = str(row[product_col]).strip() if pd.notna(row[product_col]) else ""

        if not q:
            print(f"\n⚠️ 第 {idx + 1} 列問題為空，跳過")
            fail_count += 1
            continue

        if not product_id:
            print(f"\n⚠️ 第 {idx + 1} 列產品代號為空，跳過")
            fail_count += 1
            continue

        print("\n" + "=" * 80)
        print(f"🚀 開始處理第 {idx + 1} 題")
        print(f"🏷️ 產品：{product_id}")
        print(f"📝 問題：{q}")
        print("=" * 80)

        state: GraphState = {
            "row_index": idx + 1,
            "product_id": product_id,
            "query_original": q,
        }

        try:
            app.invoke(state)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"❌ 第 {idx + 1} 題執行失敗：{e}")

    print("\n================= Batch Done =================")
    print(f"✅ 成功：{success_count}")
    print(f"❌ 失敗：{fail_count}")
    print(f"📘 結果已累積寫入：{RESULT_EXCEL_PATH}")
    print("==============================================\n")

# ---------------------------
# CLI
# ---------------------------
def main():
    app = build_app()

    print("請選擇模式：")
    print("1. 單題模式")
    print("2. 批次模式（讀取 data/題目.xlsx）")

    mode = input("請輸入 1 或 2：").strip()

    if mode == "1":
        run_single_mode(app)
    elif mode == "2":
        run_batch_mode(app)
    else:
        print("❌ 無效選項")

if __name__ == "__main__":
    main()

