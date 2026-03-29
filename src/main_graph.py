# src/main_graph.py
# LangGraph版：coverage branch
# 流程：
# query
# -> difficulty estimator
# -> hard: information analyzer
# -> easy / hard different retrieval strategies
# -> answer generate
# -> answer evaluation (coverage only; no factual)
# -> output or reAct
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

from utils import pack_docs

from tools import (
    information_need_tool,
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 專案根目錄
BATCH_QUESTION_PATH = os.path.join(PROJECT_ROOT, "data", "題目.xlsx")
RESULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, "rag_results_coverage.xlsx")


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

    # InfoNeed (hard only)
    info_needs: Any                         # information_need_tool 輸出（list[dict] 或 raw）
    
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
    must_failed: bool                       # coverage 是否有 must-have < threshold
    low_keypoints: List[Dict[str, Any]]     # must-have 低分 keypoints（points 裡挑出來的）

    # reAct Evidence Accumulation / Flags
    prev_expanded_docs: List[Dict[str, Any]]  # 上一輪（或累積後）的 expanded_docs，用於合併證據
    is_react_retrieval: bool                  # 這一輪 retrieve 是否用 reAct 規格(top10/rerank5/expand)
    react_after_retrieve: bool                # retrieve 後是否直接走 revise（跳過 generate）


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
    state["round"] = 0
    state["max_rounds"] = state.get("max_rounds", 2)

    # reAct 控制旗標
    state["react_after_retrieve"] = False
    state["is_react_retrieval"] = False
    state["prev_expanded_docs"] = []

    # 清空檢索 / 生成 / 評估結果
    state["context"] = ""
    state["expanded_docs"] = []
    state["answer"] = ""
    state["eval"] = {}
    state["eval_raw"] = ""
    state["confidence"] = 0.0
    state["must_failed"] = False
    state["low_keypoints"] = []

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

def n_evaluate(state: GraphState) -> GraphState:
    # 評估目前答案的品質（coverage + factuality），並決定是否需要 reAct 修正

    difficulty = state.get("difficulty", "easy")

    # easy 不用 info_need：直接傳空，tool 內會自動加 query 當 must-have
    info_points = [] if difficulty == "easy" else state.get("info_needs", [])

    raw = evaluate_answer_metrics(
        query=state["query_original"],   
        information_points=info_points,
        answer=state.get("answer", ""),
        expanded_docs=state.get("expanded_docs", []),
        difficulty=difficulty
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
    
    # coverage: 找出 must_have 且 分數 < 0.8 的 keypoints
    low_kps = []
    points = obj.get("points", []) or []

    if isinstance(points, list):
        for p in points:
            try:
                if p.get("must_have", True) and float(p.get("分數", 0.0)) < 0.8:
                    low_kps.append(p)
            except Exception:
                continue
    must_failed = len(low_kps) > 0

    # write back to state
    state["low_keypoints"] = low_kps
    state["must_failed"] = must_failed
        
    print("\n========== [Evaluation Result] ==========")
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    print("\n========== [Eval Summary] ==========")
    print(f"must_failed: {must_failed}")
    print(f"low_keypoints_count: {len(low_kps)}")

    return state

def n_prepare_react_query(state: GraphState) -> GraphState:
    """
    coverage failed 時才會走到這裡：
    - easy: query = original query
    - hard: query = 低分 keypoints（1個就用那個，多個就 concat）
    """

    # 啟動 reAct retrieval 並增加 round
    state["react_after_retrieve"] = True
    state["is_react_retrieval"] = True

    difficulty = state.get("difficulty", "easy")
    low_kps = state.get("low_keypoints", []) or []

    # easy：直接用原始 query
    if difficulty == "easy":
        state["query_current"] = state["query_original"]

        print("\n========== [reAct Triggered] ==========")
        print("reason: coverage failed")
        print(f"round: {state['round']}")
        print(f"new query: {state['query_current']}")

        return _log(state, f"🔁 reAct(coverage) easy：query=original")
    
    # hard：使用低分 keypoints
    if not low_kps:
        # 理論上不會發生（因為只有 coverage failed 才會進來）
        state["query_current"] = state["query_original"]

        print("\n========== [reAct Triggered] ==========")
        print("reason: coverage failed")
        print(f"round: {state['round']}")
        print(f"new query: {state['query_current']}")

        return _log(state, f"🔁 reAct(coverage) hard：low_keypoints 空，fallback original")

    # point 欄位：你 evaluator points 裡是 "point"
    kp_texts = []
    for p in low_kps:
        t = str(p.get("point", "")).strip()
        if t:
            kp_texts.append(t)

    # 只有一個 keypoint
    if len(kp_texts) == 1:
        state["query_current"] = kp_texts[0]

        print("\n========== [reAct Triggered] ==========")
        print("reason: coverage failed")
        print(f"round: {state['round']}")
        print("low_keypoints:")
        print(json.dumps(low_kps, ensure_ascii=False, indent=2))
        print(f"new query: {state['query_current']}")

        return _log(state, f"🔁 reAct(coverage) hard：query=低分keypoint(1)")

    # 多個 keypoints → 合併成一個 query
    q = "；".join(kp_texts)  # 你要用空白/逗號也行
    state["query_current"] = q

    print("\n========== [reAct Triggered] ==========")
    print("reason: coverage failed")
    print(f"round: {state['round']}")
    print("low_keypoints:")
    print(json.dumps(low_kps, ensure_ascii=False, indent=2))
    print(f"new query: {state['query_current']}")

    return _log(state, f"🔁 reAct(coverage) hard：query=低分keypoints({len(kp_texts)})")

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

    low_kps = state.get("low_keypoints", []) or []

    _log(state, f"✍️ revise_answer：coverage_only | low_kps={len(low_kps)}")

    # 取得舊 + 新文件
    old_docs = state.get("prev_expanded_docs", []) or []
    new_docs = state.get("expanded_docs", []) or []

    # 用 CHUNK_ID 去重合併
    merged = {}
    no_id_docs = []

    for d in old_docs + new_docs:
        meta = d.get("meta", {}) or {}
        cid = meta.get("CHUNK_ID")
        if cid is None:
            no_id_docs.append(d)
        elif cid not in merged:
            merged[cid] = d

    merged_docs = list(merged.values()) + no_id_docs

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
        weakness_type=["coverage"],
        low_keypoints=low_kps,
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
    state["react_after_retrieve"] = False

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

    print("\n【評估】")
    ev = state.get("eval", {}) or {}
    print(json.dumps(ev, ensure_ascii=False, indent=2))
    print(f"\n⏱️ 本題耗時：{state.get('latency')} 秒")
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

def r_after_retrieve_process(state: GraphState) -> str:
    # retrieve 完之後的分流：
    #
    # 如果這一輪是 coverage reAct 檢索
    # → 不需要重新 generate（因為已經有舊答案）
    # → 直接進 revise_answer 修正答案
    #
    # 如果是正常 retrieval
    # → 進 generate 產生答案
    return "revise_answer" if state.get("react_after_retrieve") else "generate"

def r_after_eval(state: GraphState) -> str:
    # 根據 evaluator 結果決定下一步：
    # coverage failed → 需要重新檢索（reAct retrieval）
    # factual failed（但 coverage OK）→ 不需要檢索，直接 revise_answer
    # 都 OK → finalize
    must_failed = bool(state.get("must_failed", False))

    # 若都沒問題，直接結束
    if not must_failed:
        return "finalize"

    # 只要還需要修正，就先檢查是否已達上限
    if int(state.get("round", 0)) >= int(state.get("max_rounds", 2)):
        print("答案可能不完全可信")
        return "finalize"

    # 3) ok
    return "prepare_react_query"


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
    g.add_node("evaluate", n_evaluate)     
    g.add_node("finalize", n_finalize)
    g.add_node("prepare_react_query", n_prepare_react_query)
    g.add_node("revise_answer", n_revise_answer)

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

    # Retrieve 後分流
    # 正常流程 → generate
    # coverage reAct retrieval 後 → revise_answer
    g.add_conditional_edges(
        "retrieve_process",
        r_after_retrieve_process,
        {"generate": "generate", "revise_answer": "revise_answer"}
    )
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
            "prepare_react_query": "prepare_react_query",
            "finalize": "finalize",
        }
    )

    # coverage reAct：先產生新 query，再重新檢索
    g.add_edge("prepare_react_query", "retrieve_process")

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
            "max_rounds": 2,
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
            "max_rounds": 2,
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