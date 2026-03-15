# src/main_graph.py
# Vanilla RAG version
# 流程：query -> retrieve -> generate -> finalize

import os
from typing import TypedDict, Optional, Dict, Any, List

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
# Nodes
# ---------------------------
def n_init(state: GraphState) -> GraphState:
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    state["logs"] = []
    state["retrieved"] = {}
    state["graded"] = {}

    pid = state.get("product_id")
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
# CLI
# ---------------------------
def main():
    app = build_app()

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

if __name__ == "__main__":
    main()