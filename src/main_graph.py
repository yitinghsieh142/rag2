# src/main_graph.py
# LangGraph版：依照你的流程圖（綠=easy, 紅=hard）
# - easy：Difficulty -> Semantic+Keyword(僅語意為主) -> Rerank+Expand -> Answer -> Eval -> END
# - hard：Difficulty -> InfoNeed -> HybridRetriever(語意/關鍵字) -> Rerank+Expand -> Answer -> Eval
#         -> 若 must-have coverage < threshold：Rewrite(摘要+擴寫+建議semantic/keyword) -> 回到 HybridRetriever（最多3輪）
#
# 你現有 tools.py / utils.py 幾乎不動，只是把「控制流」搬到 LangGraph。
#
# 需求：
# pip install langgraph
#
# 執行：
# cd src
# python3 main_graph.py

import os
import json
from typing import TypedDict, Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document

from langgraph.graph import StateGraph, START, END

from tools import (
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
    product_id: Optional[str]               # 產品代號/名稱（用來 build vectorstore）
    difficulty: Literal["easy", "hard"]     # difficulty predictor 結果

    # Retrieval / Docs (after retrieve_process_tool)
    retrieved: Dict[str, Any]               # 原始檢索結果（可選，用於 debug/分析）
    expanded_docs: List[Dict[str, Any]]     # expand + cap 後的文件（list[{"text","meta"}]）
    context: str                            # expanded_docs 拼接後的文字（給 LLM 生成用）

    # Generation
    answer: str                             # generate / revise 後的答案（主要輸出）

    # Debug / Logs
    logs: List[str]                           # debug 用 log list

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

    # 清空檢索 / 生成 / 評估結果
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    state["retrieved"] = {}

    # 重置 logs（避免跨 query 累積）
    state["logs"] = []

    pid = state.get("product_id")
    return _log(state, f"📌 已鎖定產品名稱：{pid}")

def n_difficulty(state: GraphState) -> GraphState:
    # 使用 difficulty predictor 判斷問題難易度（決定後續檢索策略）
    d = predict_difficulty(state["query_original"])
    state["difficulty"] = d
    return _log(state, f"🧠 難易度判斷：{d}")

def n_retrieve_process(state: GraphState) -> GraphState:
    # 執行檢索流程（依 difficulty / reAct 狀態決定 retrieve 參數）
    q = state["query_original"]

    # 呼叫 retrieve_process_tool（內部會做 retrieve → rerank → expand → pack）
    out = retrieve_process_tool(
        query=q,
        product_id=state.get("product_id"),
        difficulty=state.get("difficulty", "easy"),
        threshold=0.3,
    )

    # 把檢索結果寫回 state
    state["context"] = out.get("context", "") # 給 LLM 生成的文字 context
    state["retrieved"] = {"docs": out.get("retrieved_docs", [])} # 原始檢索文件
    state["expanded_docs"] = out.get("expanded_docs", []) # expand 後文件（給合併用）

    return state

def n_generate(state: GraphState) -> GraphState:
    # 使用目前的 context 生成答案
    out = generate_answer_tool(
        query=state["query_original"],
        context=state.get("context", ""),
    )

    state["answer"] = out.get("answer", "")

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
# Build Graph
# ---------------------------
def build_app():
    # 建立 LangGraph workflow
    g = StateGraph(GraphState)

    # nodes
    g.add_node("init", n_init)
    g.add_node("predict_difficulty", n_difficulty)

    # 共用流程：檢索 / 生成
    g.add_node("retrieve_process", n_retrieve_process)
    g.add_node("generate", n_generate)     
    g.add_node("finalize", n_finalize)

    g.add_edge(START, "init")
    g.add_edge("init", "predict_difficulty")
    g.add_edge("predict_difficulty", "retrieve_process")
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