# src/main_graph.py
# IA版本：Vanilla RAG + Difficulty Estimator + Information Need Analyzer
#
# 流程：
# - easy：Query -> Difficulty -> Retrieve -> Generate -> END
# - hard：Query -> Difficulty -> InfoNeed -> Retrieve -> Generate -> END
#
# 說明：
# - 沒有 evaluator
# - 沒有 revise / rewrite / reAct loop
# - information need analyzer 只在 hard 題啟動
# - analyzer 的輸出只會作為 prompt 補充資訊傳給 generate_answer_tool

import os
import json
from typing import TypedDict, Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document

from langgraph.graph import StateGraph, START, END

from tools import (
    information_need_tool,
    retrieve_process_tool,
    generate_answer_tool,
)
from difficulty import predict_difficulty
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
    # User Input / Query
    query_original: str                     # 使用者原始問題（整個流程不應改動）
    query_current: str                      # 當前要拿去檢索/生成用的 query（reAct 時可能變）
    product_id: Optional[str]               # 產品代號/名稱（用來 build vectorstore）
    difficulty: Literal["easy", "hard"]     # difficulty predictor 結果

    # InfoNeed (hard only)
    info_needs: Any                         # information_need_tool 輸出（list[dict] 或 raw）
    
    # Retrieval / Docs (after retrieve_process_tool)
    retrieved: Dict[str, Any]               # 原始檢索結果（可選，用於 debug/分析）
    graded: Dict[str, Any]                  # retrieve_process_tool 回傳整包（你目前用 graded 存）
    expanded_docs: List[Dict[str, Any]]     # expand + cap 後的文件（list[{"text","meta"}]）
    context: str                            # expanded_docs 拼接後的文字（給 LLM 生成用）

    # Generation
    answer: str                             # generate / revise 後的答案（主要輸出）

    # Debug / Logs
    logs: List[str]                           # debug 用 log list
    answer_history: List[str]

# ---------------------------
# 小工具：log / parse
# ---------------------------
def _log(state: GraphState, msg: str) -> GraphState:
    # 將訊息加入 state["logs"] 並同時印出（用於 debug）
    logs = state.get("logs", [])
    logs.append(msg)
    state["logs"] = logs
    print(msg)
    return state

# ---------------------------
# Nodes
# ---------------------------
def n_init(state: GraphState) -> GraphState:
    # 初始化整個流程的狀態，重置所有控制與評估相關欄位
    q = state["query_original"]

    state["query_current"] = q
    state["info_needs"] = []
    
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    
    # 重置 logs（避免跨 query 累積）
    state["logs"] = []
    state["answer_history"] = []

    pid = state.get("product_id")
    return _log(state, f"📌 已鎖定產品名稱：{pid}")

def n_difficulty(state: GraphState) -> GraphState:
    # 使用 difficulty predictor 判斷問題難易度（決定後續檢索策略）
    d = predict_difficulty(state["query_original"])
    state["difficulty"] = d
    return _log(state, f"🧠 難易度判斷：{d}")

def n_info_need(state: GraphState) -> GraphState:
    # 針對 hard 問題拆解成 must/opt 知識點（easy 會略過這一步）
    out = information_need_tool(state["query_current"])
    state["info_needs"] = out.get("info_needs", out)
    return state

def n_retrieve_process(state: GraphState) -> GraphState:
    # 執行檢索流程（依 difficulty / reAct 狀態決定 retrieve 參數）
    
    # 取得目前 query
    q = state.get("query_current", state["query_original"])

    # 呼叫 retrieve_process_tool（內部會做 retrieve → rerank → expand → pack）
    out = retrieve_process_tool(
        query=q,
        product_id=state.get("product_id"),
        difficulty=state.get("difficulty", "easy"),
        threshold=0.3,
    )

    # 把檢索結果寫回 state
    state["graded"] = out # 整包輸出（debug 用）
    state["context"] = out.get("context", "") # 給 LLM 生成的文字 context
    state["retrieved"] = {"docs": out.get("retrieved_docs", [])} # 原始檢索文件
    state["expanded_docs"] = out.get("expanded_docs", []) # expand 後文件（給合併用）

    return state

def n_generate(state: GraphState) -> GraphState:
    # 使用目前的 context 生成答案
    difficulty = state.get("difficulty", "easy")

    info_points = []
    if difficulty == "hard":
        info_points = state.get("info_needs", []) or []

    out = generate_answer_tool(
        query=state["query_current"],
        context=state.get("context", ""),
        info_points=info_points
    )

    state["answer"] = out.get("answer", "")
    if state["answer"]:
        history = state.get("answer_history", [])
        history.append(state["answer"])
        state["answer_history"] = history
    print("\n========== [Generated Answer] ==========")
    print(state["answer"])
    return state


def n_finalize(state: GraphState) -> GraphState:
    # 輸出最終答案、來源 chunk 與評估結果（目前用 print，之後可改成回傳 JSON）
    print("\n================= Final Response =================")
    print("【答案】")
    print(state.get("answer", ""))

    print("\n【來源（Top context chunks）】")
    # grade_documents_tool 回傳 expanded_docs（packed["docs"]），這裡拿來印 chunk_id
    expanded = state.get("expanded_docs", [])
    if isinstance(expanded, list) and expanded:
        for i, d in enumerate(expanded[:5]):
            cid = (d.get("meta", {}) or {}).get("CHUNK_ID")
            title = (d.get("meta", {}) or {}).get("TITLE", "")
            print(f"- [{i}] CHUNK_ID={cid} {title}".strip())
    else:
        print("(無)")

    print("=====================================================\n")
    excel_path = append_graph_result_to_excel(
        state,
        excel_path="rag_results.xlsx"
    )
    print(f"📘 已寫入 Excel：{excel_path}")
    return state

# ---------------------------
# Routers (conditional edges)
# ---------------------------
def r_after_difficulty(state: GraphState) -> str:
    # 根據 difficulty 決定流程分支：
    # easy → 不需要 information_need
    # hard → 需要先拆解 key information points
    return "easy" if state.get("difficulty") == "easy" else "hard"

# ---------------------------
# Build Graph
# ---------------------------
def build_app():
    # 建立 LangGraph workflow
    g = StateGraph(GraphState)

    # nodes
    g.add_node("init", n_init)
    g.add_node("predict_difficulty", n_difficulty)

    # hard 專用：先拆解 information needs
    g.add_node("info_need", n_info_need)

    # 共用流程：檢索 / 生成 / 評估 / 修正 / 輸出
    g.add_node("retrieve_process", n_retrieve_process)
    g.add_node("generate", n_generate)
    g.add_node("finalize", n_finalize)

    # 起點流程
    g.add_edge(START, "init")
    g.add_edge("init", "predict_difficulty")

    # Difficulty 分流
    # easy → 直接檢索
    # hard → 先做 information need 再檢索
    g.add_conditional_edges(
        "predict_difficulty",
        r_after_difficulty,
        {"easy": "retrieve_process", "hard": "info_need"}
    )

    # hard chain
    g.add_edge("info_need", "retrieve_process")
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

        # 每次 query 初始化 state
        state: GraphState = {
            "product_id": product_id,
            "query_original": q,
            "max_rounds": 2,
        }

        print("🔍 正在分析並回答問題...")
        app.invoke(state)

if __name__ == "__main__":
    main()