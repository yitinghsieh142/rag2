# src/main_graph.py
# LangGraph版：reranker branch
#
# 流程：
# query -> difficulty estimator -> retrieve_process -> generate -> finalize
#
# 規則：
# - easy：top-k 較小、不 rerank、expand 較少
# - hard：top-k 較大、要 rerank、expand 較多
#
# 注意：
# - 此版本不包含 info analyzer
# - 此版本不包含 evaluator
# - 此版本不包含 reAct / revise loop
# - retrieve_process_tool 內部負責依 difficulty 決定：
#   retrieve -> (optional rerank) -> expand -> pack
#
# 執行：
# cd src
# python3 main_graph.py

import os
import json
import time
from typing import TypedDict, Optional, List, Dict, Any, Literal
import pandas as pd
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 專案根目錄
BATCH_QUESTION_PATH = os.path.join(PROJECT_ROOT, "data", "題目.xlsx")
RESULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, "rag_results_reranker.xlsx")


# ---------------------------
# Graph State
# ---------------------------
class GraphState(TypedDict, total=False):
    # User Input / Query
    query_original: str                     # 使用者原始問題（整個流程不應改動）
    query_current: str                      # 當前要拿去檢索/生成用的 query（reAct 時可能變）
    product_id: Optional[str]               # 產品代號/名稱（用來 build vectorstore）
    difficulty: Literal["easy", "hard"]     # difficulty predictor 結果

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

    # batch / timing
    row_index: int
    start_time: float
    latency: float

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
    
    # 初始化整個流程的狀態，重置所有控制與評估相關欄位
    q = state["query_original"]

    # 基本 query 狀態
    state["query_current"] = q

    # 清空檢索 / 生成 / 評估結果
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""

    # 重置 logs（避免跨 query 累積）
    state["logs"] = []
    state["answer_history"] = []

    pid = state.get("product_id")
    row_index = state.get("row_index")

    if row_index is not None:
        return _log(state, f"📌 [第 {row_index} 題] 已鎖定產品名稱：{pid}")
    
    return _log(state, f"📌 已鎖定產品名稱：{pid}")

def n_difficulty(state: GraphState) -> GraphState:
    # 使用 difficulty predictor 判斷問題難易度（決定後續檢索策略）
    d = predict_difficulty(state["query_original"])
    state["difficulty"] = d
    return _log(state, f"🧠 難易度判斷：{d}")

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
    out = generate_answer_tool(
        query=state["query_current"],
        context=state.get("context", "")
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
    start_time = state.get("start_time")
    if start_time:
        state["latency"] = round(time.time() - start_time, 2)
    else:
        state["latency"] = None

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

    print(f"\n⏱️ 本題耗時：{state.get('latency')} 秒")

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

    # 共用流程：檢索 / 生成 / 評估 / 修正 / 輸出
    g.add_node("retrieve_process", n_retrieve_process)
    g.add_node("generate", n_generate)    
    g.add_node("finalize", n_finalize)

    # 起點流程
    g.add_edge(START, "init")
    g.add_edge("init", "predict_difficulty")

    g.add_edge("predict_difficulty", "retrieve_process")

    # hard chain
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

        print("🔍 正在分析並回答問題...")
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
    print(f"📑 欄位：{list(df.columns)}")

    question_col = find_column(df, ["問題", "query", "question", "題目"])
    product_col = find_column(df, ["產品代號", "product_id", "product", "商品代號", "商品名稱", "產品名稱"])

    if question_col is None:
        print("❌ 題目.xlsx 找不到問題欄位")
        print("   可接受欄名：問題 / query / question / 題目")
        return

    if product_col is None:
        print("❌ 題目.xlsx 找不到產品欄位")
        print("   可接受欄名：產品代號 / product_id / product / 商品代號 / 商品名稱 / 產品名稱")
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
    print("2. 批次模式（讀取 ../data/題目.xlsx）")

    mode = input("請輸入 1 或 2：").strip()

    if mode == "1":
        run_single_mode(app)
    elif mode == "2":
        run_batch_mode(app)
    else:
        print("❌ 無效選項")

if __name__ == "__main__":
    main()
