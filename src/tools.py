from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable
from langchain.schema import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from FlagEmbedding import FlagReranker


import os
import re 
import json
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple, Optional, Literal

from utils import (
    extract_prod_id_from_query,
    build_vectorstore,
    expand_retrieved_chunks_v2,
    FilteredRetriever,
    pack_docs,
)

from prompt import prompt, answer_evaluation_prompt, query_expanding_prompt, information_need_prompt, answer_revision_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === 讀取環境變數 ===
load_dotenv()

client = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"trust_remote_code": True, "device": "mps"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_FP16 = os.getenv("RERANKER_FP16", "1").lower() in {"1", "true", "yes"}
reranker = FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_FP16)


# === 各指標答案評分 ===
def evaluate_answer_metrics(
    query: str,
    information_points,
    answer: str,
    expanded_docs,
    context: str = "",
    difficulty: str = "hard",
) -> str:
    """
    Answer Evaluator (Coverage only)

    Coverage:
      - 由 answer_evaluation_prompt (LLM) 針對 info points 打分 (0~1)
      - must_have=True 權重 0.8；False 權重 0.2
      - 得到 coverage_score ∈ [0,1]

    回傳 JSON 字串：
    {
      "points": [...],
      "coverage_score": 0.xx,
      "C_conf": ...,
      "信心分數": ...
    }
    """

    # -----------------------
    # helpers
    # -----------------------
    def _normalize_info_points(obj):
        if isinstance(obj, str):
            s = obj.strip()
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    start, end = s.find('['), s.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        obj = json.loads(s[start:end+1])
                    else:
                        obj = [ln.strip() for ln in s.splitlines() if ln.strip()]
                except Exception:
                    obj = [ln.strip() for ln in s.splitlines() if ln.strip()]

        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return [{"id": str(i), "description": desc.strip(), "must_have": True}
                    for i, desc in enumerate(obj, start=1)]

        norm = []
        if isinstance(obj, list):
            seen = set()
            for i, item in enumerate(obj, start=1):
                if not isinstance(item, dict):
                    continue
                _id = str(item.get("id", str(i)))
                desc = str(item.get("description", "")).strip()
                if not desc or desc in seen:
                    continue
                must_raw = item.get("must_have", True)
                must = (must_raw if isinstance(must_raw, bool)
                        else str(must_raw).lower() in {"true", "1", "yes", "y"})
                norm.append({"id": _id, "description": desc, "must_have": must})
                seen.add(desc)
        return norm
    
    def _get_doc_fields(d):
        if isinstance(d, dict):
            meta = d.get("meta", {}) or {}
            title = meta.get("TITLE", "")
            text = d.get("text", "") or ""
            return title, text
        else:
            meta = getattr(d, "metadata", {}) or {}
            title = meta.get("TITLE", "")
            text = getattr(d, "page_content", "") or ""
            return title, text

    # 1) easy: 把 query 本身當作 must-have keypoint（不需經過 information_need_tool）

    norm_points = _normalize_info_points(information_points)

    if difficulty.lower() == "easy":
        norm_points = [{
            "id": "Q",
            "description": query.strip(),
            "must_have": True
        }]

    # 最多 5 個（避免 prompt 爆）
    norm_points = norm_points[:5]
    info_points_json = json.dumps(norm_points, ensure_ascii=False)

    # 2) Coverage by LLM
    chain = answer_evaluation_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "information_points": info_points_json,
        "answer": answer
    })
    raw = (result_msg.content or "").strip()

    # model_points = _parse_points(raw)
    chain = answer_evaluation_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "information_points": info_points_json,
        "answer": answer
    })

    raw = (result_msg.content or "").strip()

    try:
        model_points = json.loads(raw)
    except Exception:
        model_points = []

    must_map = {p["id"]: bool(p.get("must_have", True)) for p in norm_points}

    fixed_points = []
    for p in model_points:
        pid = str(p.get("id", "")).strip()

        try:
            score = float(p.get("分數", 0.0))
        except Exception:
            score = 0.0

        fixed_points.append({
            "id": pid,
            "point": p.get("point", ""),
            "must_have": must_map.get(pid, True),
            "分數": max(0.0, min(1.0, score))
        })

    # 若模型輸出異常，fallback：用 norm_points 建空白分數
    if not fixed_points:
        fixed_points = [{
            "id": p["id"], "point": p["description"], "must_have": p["must_have"], "分數": 0.0
        } for p in norm_points]

    # 5) 計算「信心分數」（加權平均）
    total_w = 0.0
    total_ws = 0.0

    for p in fixed_points:
        w = 0.8 if p.get("must_have", True) else 0.2
        s = float(p.get("分數", 0.0))
        total_w += w
        total_ws += w * s
    # confidence = round((total_ws / total_w) if total_w > 0 else 0.0, 4)
    coverage_score = round(total_ws / total_w, 4) if total_w else 0.0
    C_conf = coverage_score

    output = {
        "points": fixed_points,
        "coverage_score": coverage_score,
        "C_conf": C_conf
    }

    return json.dumps(output, ensure_ascii=False)


# === information Need Tool（將問題拆成知識點清單） ===
def information_need_tool(query: str) -> dict:
    """
    依據使用者問題，產出回答該問題所需的 Information Needs 清單。
    會呼叫 information_need_prompt（LLM）→ 解析純 JSON → 清洗。
    return: {"info_needs": [{"id": "...", "description": "...", "must_have": True/False}, ...]}
    """
    # 1) 呼叫 LLM 取得原始輸出（預期是純 JSON 陣列字串）
    chain = information_need_prompt | client
    result_msg = chain.invoke({"query": query})
    raw_text = getattr(result_msg, "content", str(result_msg)).strip()

    # 2) 嘗試解析 JSON；若模型前後多了雜訊，容錯擷取最外層的 [] 片段
    def _parse_json_array(s: str):
        try:
            return json.loads(s)
        except Exception:
            # 抓出第一個 '[' 到最後一個 ']'，嘗試重新解析
            try:
                start = s.find('[')
                end = s.rfind(']')
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start:end+1])
            except Exception:
                pass
        return []

    parsed = _parse_json_array(raw_text)
    if not isinstance(parsed, list):
        parsed = []

    # 3) 正規化欄位與型別、去重（以 description 為 key）
    cleaned = []
    seen_desc = set()
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        _id = str(item.get("id", str(i)))
        desc = str(item.get("description", "")).strip()
        if not desc or desc in seen_desc:
            continue
        must_have_raw = item.get("must_have", True)
        must_have = bool(must_have_raw) if isinstance(must_have_raw, bool) else str(must_have_raw).lower() in {"true", "1", "yes", "y"}
        cleaned.append({
            "id": _id,
            "description": desc,
            "must_have": must_have
        })
        seen_desc.add(desc)

    # 最多保留 8 個，避免過長
    if len(cleaned) > 8:
        cleaned = cleaned[:8]

    # Debug 印出
    print("\n========== [Information Needs] ==========")
    for n in cleaned:
        print(f"- ({'MUST' if n['must_have'] else 'OPT'}) {n['id']}: {n['description']}")

    return {"info_needs": cleaned}

#  ===  語意檢索  === 
Difficulty = Literal["easy", "hard"]
def retrieve_process_tool(
    query: str,
    product_id: Optional[str] = None,
    difficulty: Difficulty = "easy",
    threshold: float = 0.3,
    is_react: bool = False,
    # 初始檢索規格
    k_retrieve_easy: int = 5,
    k_retrieve_hard: int = 8,
    # reAct 檢索規格
    k_retrieve_react: int = 10,
    top_k_rerank: int = 5,
    # expand cap
    max_expand_easy: int = 10,
    max_expand_hard: int = 15,
) -> dict:
    """
    單一入口：依 difficulty 決定流程
    - easy: retrieve 5 -> no rerank -> expand -> cap 10
    - hard: retrieve 8 -> rerank(top_k=5) -> expand -> cap 15

    回傳格式（統一）：
    {
      "retrieved_docs": [...],     # raw retrieved（pack_docs 的 docs）
      "scored_docs": [...],        # rerank 後 top docs（easy 空）
      "expanded_docs": [...],      # 最終餵給 LLM 的 docs（cap 後）
      "context": "..."             # expanded_docs 拼接的 context
    }
    """
    # note: 可以刪掉 scored_docs

    # -----------------------
    # 0) decide params
    # -----------------------
    # if difficulty == "easy":
    #     k_retrieve = k_retrieve_easy
    #     max_expand = max_expand_easy
    #     do_rerank = False
    #     top_k_rerank = 0
    # else:
    #     k_retrieve = k_retrieve_hard
    #     max_expand = max_expand_hard
    #     do_rerank = True
    #     top_k_rerank = top_k_rerank_hard
    if is_react:
        # reAct 規格（你指定）
        k_retrieve = k_retrieve_react
        do_rerank = True
        max_expand = max_expand_hard
    else:
        # 初始檢索（維持你原本規格）
        if difficulty == "easy":
            k_retrieve = k_retrieve_easy       # 5
            max_expand = max_expand_easy       # 10
            do_rerank = False
        else:
            k_retrieve = k_retrieve_hard       # 8
            max_expand = max_expand_hard       # 15
            do_rerank = True

    prod_id = product_id or extract_prod_id_from_query(query)
    if not prod_id:
        return {"error": "❌ 無法判斷產品名稱"}

    vectorstore = build_vectorstore(prod_id)

    # -----------------------
    # 1) semantic retrieve (raw)
    # -----------------------
    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=threshold, k=k_retrieve)
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        retrieved_docs = vectorstore.similarity_search(query, k=k_retrieve)

    print(f"\n========== [Semantic Retrieved Docs (raw) | {difficulty}] ==========")
    for i, d in enumerate(retrieved_docs):
        print(f"[{i}] CHUNK_ID={d.metadata.get('CHUNK_ID')}\n{d.page_content}\n")

    retrieved_packed = pack_docs(retrieved_docs, max_docs=k_retrieve)
    docs_in = retrieved_packed.get("docs", [])

    if not docs_in:
        return {
            "retrieved_docs": [],
            "expanded_docs": [],
            "context": "",
            "warning": "retrieved 空或缺少 docs"
        }

    # -----------------------
    # 2) optional rerank (hard only)
    # -----------------------
    if do_rerank:
        pairs = [[query, d["text"]] for d in docs_in]
        try:
            scores = reranker.compute_score(pairs, normalize=True)
        except Exception as e:
            return {
                "retrieved_docs": docs_in,
                "expanded_docs": [],
                "context": "",
                "error": f"reranker 失敗：{e}"
            }

        scored = []
        for d, s in zip(docs_in, scores):
            e = dict(d)
            e["score"] = float(s)
            scored.append(e)

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:max(1, min(top_k_rerank, len(scored)))]

        print("\n========== [Reranked Top Docs - BGE v2-m3 | hard] ==========")
        for i, d in enumerate(top):
            cid = d.get("meta", {}).get("CHUNK_ID")
            print(f"[{i}] score={d['score']:.3f} CHUNK_ID={cid}\n{d['text']}\n")

        base_docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in top]
    else:
        # easy: 不 rerank，直接用原順序
        base_docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in docs_in]


    # -----------------------
    # 3) expand + cap
    # -----------------------
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, base_docs)

    if len(expanded_docs) > max_expand:
        expanded_docs = expanded_docs[:max_expand] 

    packed = pack_docs(expanded_docs, max_docs=max_expand)

    print(f"\n========== [Expanded Docs | cap={max_expand} | {difficulty}] ==========")
    for i, d in enumerate(packed["docs"]):
        cid = d.get("meta", {}).get("CHUNK_ID")
        print(f"[{i}] CHUNK_ID={cid}\n{d['text']}\n")

    return {
        "retrieved_docs": docs_in,
        "expanded_docs": packed["docs"],
        "context": packed["context"],
    }


#  ===  答案生成工具  === 
def generate_answer_tool(query: str, context: str, info_points=None) -> dict:    
    """
    只負責依 context 生成答案（不做檢索與重排）
    回傳：{"answer": "...", "context": context}
    """
    ctx = (context or "").strip()
    if not ctx:
        return {"answer": "找不到相關內容", "context": ""}

    try:
        if info_points:
            info_points_str = json.dumps(info_points, ensure_ascii=False, indent=2)
        else:
            info_points_str = "[]"
    except Exception:
        info_points_str = "[]"

    chain = prompt | client
    result_msg = chain.invoke({"context": ctx, "query": query, "info_points": info_points_str})
    answer = result_msg.content.strip()
    return {"answer": answer, "context": ctx}

#  ===  修正答案生成工具  === 
def revise_answer_tool(
    query: str,
    prev_answer: str,
    weakness_type,
    low_keypoints,
    context: str
) -> dict:
    """
    重新生成答案（修正版）
    """

    # weakness_type → 轉成字串
    if isinstance(weakness_type, list):
        weakness_str = json.dumps(weakness_type, ensure_ascii=False)
    else:
        weakness_str = str(weakness_type)

    # low_keypoints → JSON 字串
    try:
        low_kp_str = json.dumps(low_keypoints or [], ensure_ascii=False)
    except Exception:
        low_kp_str = "[]"

    chain = answer_revision_prompt | client
    result_msg = chain.invoke({
        "query": (query or "").strip(),
        "prev_answer": (prev_answer or "").strip(),
        "weakness_type": weakness_str,
        "low_keypoints": low_kp_str,
        "retrieved_docs": context  # ← 已經是文字
    })

    return {"answer": result_msg.content.strip()}