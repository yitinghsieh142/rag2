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

from utils import pack_docs

from tools import (
    retrieve_process_tool,
    generate_answer_tool,
    evaluate_answer_metrics,      
    revise_answer_tool,

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

    # Global Control / Loop
    max_rounds: int                         # reAct 最多幾輪
    round: int                              # 目前第幾輪 reAct（coverage 觸發才會 +1）

    # Retrieval / Docs (after retrieve_process_tool)
    retrieved: Dict[str, Any]               # 原始檢索結果（可選，用於 debug/分析）
    graded: Dict[str, Any]                  # retrieve_process_tool 回傳整包（你目前用 graded 存）
    expanded_docs: List[Dict[str, Any]]     # expand + cap 後的文件（list[{"text","meta"}]）
    context: str                            # expanded_docs 拼接後的文字（給 LLM 生成用）

    # Generation
    answer: str                             # generate / revise 後的答案（主要輸出）

    # Evaluation (coverage + factuality)
    eval_raw: str                           # evaluator 原始輸出（JSON string）
    eval: Dict[str, Any]                    # eval_raw parse 成 dict
    confidence: float                       # evaluator 給的信心分數（你目前是 C_conf 或 信心分數）
    factual_failed: bool                    # factual 是否 < threshold
    weakness_types: List[Literal["factual"]] # 可能同時包含 coverage/factual

    # reAct Evidence Accumulation / Flags
    prev_expanded_docs: List[Dict[str, Any]]  # 上一輪（或累積後）的 expanded_docs，用於合併證據
    is_react_retrieval: bool                  # 這一輪 retrieve 是否用 reAct 規格(top10/rerank5/expand)

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

def _coerce_json_obj(s: str) -> Optional[dict]:
    # 嘗試將字串安全轉為 dict（容錯處理，支援雜訊包裹的 JSON）
    if not isinstance(s, str):
        return None
    txt = s.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try extract outermost {...}
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(txt[start:end+1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# ---------------------------
# Nodes
# ---------------------------
def n_init(state: GraphState) -> GraphState:
    # 初始化整個流程的狀態，重置所有控制與評估相關欄位
    q = state["query_original"]

    # 基本 query 狀態
    state["query_current"] = q
    state["round"] = 0
    state["max_rounds"] = state.get("max_rounds", 2)

    # reAct 控制旗標
    state["is_react_retrieval"] = False
    state["prev_expanded_docs"] = []

    # 清空檢索 / 生成 / 評估結果
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    state["eval"] = {}
    state["eval_raw"] = ""
    state["confidence"] = 0.0
    state["factual_failed"] = False
    state["weakness_types"] = []

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
        is_react=bool(state.get("is_react_retrieval", False)),
    )

    # 把檢索結果寫回 state
    state["graded"] = out # 整包輸出（debug 用）
    state["context"] = out.get("context", "") # 給 LLM 生成的文字 context
    state["retrieved"] = {"docs": out.get("retrieved_docs", [])} # 原始檢索文件
    state["expanded_docs"] = out.get("expanded_docs", []) # expand 後文件（給合併用）
    state["is_react_retrieval"] = False

    return state

def n_generate(state: GraphState) -> GraphState:
    out = generate_answer_tool(
        query=state["query_current"],
        context=state.get("context", ""),
    )

    state["answer"] = out.get("answer", "")
    if state["answer"]:
        history = state.get("answer_history", [])
        history.append(state["answer"])
        state["answer_history"] = history
    print("\n========== [Generated Answer] ==========")
    print(state["answer"])
    return state

def n_evaluate(state: GraphState) -> GraphState:
    # 評估目前答案的品質（coverage + factuality），並決定是否需要 reAct 修正

    raw = evaluate_answer_metrics(
        query=state["query_original"],   
        answer=state.get("answer", ""),
        expanded_docs=state.get("expanded_docs", []),
        difficulty=state.get("difficulty", "easy")
    )

    # 保存 evaluator 原始輸出
    state["eval_raw"] = raw
    obj = _coerce_json_obj(raw) or {}
    state["eval"] = obj

    # 信心分數
    try:
        state["confidence"] = float(obj.get("C_conf", obj.get("信心分數", 0.0)))
    except Exception:
        state["confidence"] = 0.0

    # factual: 取 nli.C_fact < 0.8
    factual_failed = False
    try:
        nli = obj.get("nli", {}) or {}
        C_fact = float(nli.get("C_fact", nli.get("factuality", 0.0)))
        factual_failed = C_fact < 0.8
    except Exception:
        factual_failed = False

    # write back to state
    state["factual_failed"] = factual_failed
    state["weakness_types"] = ["factual"] if factual_failed else []

    print("\n========== [Evaluation Result] ==========")
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    print("\n========== [Eval Summary] ==========")
    print(f"factual_failed: {factual_failed}")
    print(f"weakness_types: {state['weakness_types']}")

    return state

def n_revise_answer(state: GraphState) -> GraphState:
    # 根據舊答案與合併後的檢索證據，重新生成修正版答案
    """
    修正版答案生成流程：
    1) merge 舊 + 新 expanded_docs
    2) 去重（以 CHUNK_ID 為 key）
    3) pack 成 context
    4) 呼叫 revise_answer_tool
    """
    state["round"] = int(state.get("round", 0)) + 1

    weakness = state.get("weakness_types", []) or []
    _log(state, f"✍️ revise_answer：weakness={weakness}")

    # 取得舊 + 新文件
    old_docs = state.get("prev_expanded_docs", []) or []
    new_docs = state.get("expanded_docs", []) or []

    # 用 CHUNK_ID 去重合併
    merged = {}
    for d in old_docs + new_docs:
        meta = d.get("meta", {}) or {}
        cid = meta.get("CHUNK_ID")
        if cid is not None and cid not in merged:
            merged[cid] = d

    merged_docs = list(merged.values())

    # 若沒有文件，直接返回原答案
    if not merged_docs:
        _log(state, "⚠️ merged_docs 為空，維持原答案")
        return state

    # pack 成 context 字串（LLM 使用
    docs_for_pack = [
        Document(page_content=d["text"], metadata=d["meta"])
        for d in merged_docs
    ]

    packed = pack_docs(docs_for_pack, max_docs=15)
    merged_context = packed.get("context", "")

    # 呼叫 revise_answer_tool
    out = revise_answer_tool(
        query=state.get("query_original", ""),
        prev_answer=state.get("answer", ""),
        weakness_type=weakness,
        context=merged_context
    )

    new_answer = out.get("answer", "").strip()

    # 更新 state
    if new_answer:
        state["answer"] = new_answer
        history = state.get("answer_history", [])
        history.append(new_answer)
        state["answer_history"] = history
    
    print("\n========== [Revised Answer] ==========")
    print(new_answer)

    # merge 後的文件就是這一輪真正用來生成與後續評估的 evidence
    state["expanded_docs"] = merged_docs
    state["context"] = merged_context
    state["prev_expanded_docs"] = merged_docs

    # 關閉 reAct 相關旗標
    state["is_react_retrieval"] = False

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

    print("\n【評估】")
    ev = state.get("eval", {}) or {}
    print(json.dumps(ev, ensure_ascii=False, indent=2))
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

def r_after_eval(state: GraphState) -> str:
    factual_failed = bool(state.get("factual_failed", False))

    if not factual_failed:
        return "finalize"

    if int(state.get("round", 0)) >= int(state.get("max_rounds", 2)):
        print("答案可能不完全可信")
        return "finalize"

    return "revise_answer"


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
    g.add_node("evaluate", n_evaluate)     
    g.add_node("finalize", n_finalize)
    g.add_node("revise_answer", n_revise_answer)

    # 起點流程
    g.add_edge(START, "init")
    g.add_edge("init", "predict_difficulty")
    g.add_edge("predict_difficulty", "retrieve_process")
    g.add_edge("retrieve_process", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("revise_answer", "evaluate")

    # Evaluate 後分流
    # coverage fail → prepare_react_query → retrieve_process
    # factual fail  → revise_answer
    # 都 OK         → finalize
    g.add_conditional_edges(
        "evaluate",
        r_after_eval,
        {
            "revise_answer": "revise_answer",
            "finalize": "finalize",
        }
    )
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