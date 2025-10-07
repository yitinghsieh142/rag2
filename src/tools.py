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
from typing import List, Set, Tuple, Optional

from utils import (
    extract_prod_id_from_query,
    build_vectorstore,
    load_softlink_mapping,
    get_softlinked_appendix_titles,
    expand_retrieved_chunks_v2,
    build_retrieval_qa_chain,
    FilteredRetriever,
    keyword_based_retriever,
    pack_docs,
)

from prompt import prompt, answer_evaluation_prompt, query_expanding_prompt, information_need_prompt

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

# === tool：各指標答案評分 ===
def evaluate_answer_metrics(query: str, information_points, answer: str) -> str:
    """
    依據 InformationNeedTool 產生的資訊點評估答案覆蓋度。
    現在會「評分所有資訊點」，並在回傳中加上「信心分數」：
    - 權重：must_have=True → 0.8；must_have=False → 0.2
    - 計算：加權平均 = sum(weight_i * score_i) / sum(weight_i)
    支援的 information_points 型態：
      - list[dict], list[str], str(JSON)
    回傳：JSON 字串：
    {
      "points": [ { "id","point","must_have","分數" }, ... ],
      "信心分數": <float>
    }
    """

    # 1) 正規化資訊點
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

    norm_points = _normalize_info_points(information_points)

    # 限制 1~8 條，避免過長（但評分是全部，不再只取 must_have）
    if len(norm_points) > 8:
        norm_points = norm_points[:8]

    # 2) 丟給評分 Prompt（新版不吃 context；要求輸出每點含 must_have）
    info_points_json = json.dumps(norm_points, ensure_ascii=False)
    chain = answer_evaluation_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "information_points": info_points_json,
        "answer": answer
    })
    raw = (result_msg.content or "").strip()

    # 3) 嘗試解析模型輸出的 JSON array
    def _parse_points(s: str):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return arr
        except Exception:
            # 寬鬆抓取最外層 []
            try:
                start, end = s.find('['), s.rfind(']')
                if start != -1 and end != -1 and end > start:
                    arr = json.loads(s[start:end+1])
                    if isinstance(arr, list):
                        return arr
            except Exception:
                pass
        return []

    model_points = _parse_points(raw)

    # 4) 保障 must_have 正確性（以 upstream 的 norm_points 為基準回填）
    must_map = {p["id"]: bool(p.get("must_have", True)) for p in norm_points}
    fixed_points = []
    for p in model_points:
        pid = str(p.get("id", "")).strip()
        point_desc = p.get("point") or p.get("description") or ""
        score = p.get("分數")
        # 清洗
        try:
            score = float(score)
        except Exception:
            score = 0.0
        fixed_points.append({
            "id": pid if pid else str(len(fixed_points) + 1),
            "point": str(point_desc).strip(),
            "must_have": must_map.get(pid, True),
            "分數": max(0.0, min(1.0, score)),
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
    confidence = round((total_ws / total_w) if total_w > 0 else 0.0, 4)

    # 6) 回傳整包 JSON
    out = {
        "points": fixed_points,
        "信心分數": confidence
    }
    return json.dumps(out, ensure_ascii=False)

# === tool #4：修改 query ===
def query_expanding_metrics(query: str, context: str, answer: str) -> str:
    chain = query_expanding_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "context": context,
        "answer": answer
    })
    return result_msg.content.strip()

def keyword_retriever_tool(query: str, keywords: List[str], k: int = 8) -> dict:
    """
    用 keyword_based_retriever（BM25）做關鍵字檢索。僅回傳前 k 篇原始命中文件（不 expand）。
    """
    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        return {"error": "❌ 無法判斷產品名稱"}

    vectorstore = build_vectorstore(prod_id)
    matched_docs = keyword_based_retriever(vectorstore, query, keywords, top_k=k)

    if not matched_docs:
        return {"warning": "⚠️ 關鍵字檢索未命中任何條文內容", "docs": [], "context": ""}

    print("\n========== [Keyword Retrieved Docs (raw)] ==========")
    for i, d in enumerate(matched_docs):
        print(f"[{i}] CHUNK_ID={d.metadata.get('CHUNK_ID')}\n{d.page_content}\n")

    return pack_docs(matched_docs, max_docs=k)

# === tool #0：Information Need Tool（將問題拆成知識點清單） ===
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

# --- 新增：語意檢索（只檢索，不 expand、不生成） ---
def semantic_retriever_tool(query: str, threshold: float = 0.3, k: int = 8) -> dict:
    """
    用 FilteredRetriever 做語意檢索。僅回傳前 k 篇原始命中文件（不 expand）。
    回傳格式：
    {
      "docs": [{"doc_id","text","meta"}, ...],   # 命中文件清單
      "context": "text1\n---\ntext2..."          # 直接拼接（僅供檢視；後續會重排）
    }
    """
    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        return {"error": "❌ 無法判斷產品名稱"}

    vectorstore = build_vectorstore(prod_id)
    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=threshold, k=k)
    retrieved_docs = retriever.invoke(query)  # List[Document]，原始命中

    print("\n========== [Semantic Retrieved Docs (raw)] ==========")
    for i, d in enumerate(retrieved_docs):
        print(f"[{i}] CHUNK_ID={d.metadata.get('CHUNK_ID')}\n{d.page_content}\n")

    return pack_docs(retrieved_docs, max_docs=k)


# --- 新增：重排工具（取 3 篇，再 expand） ---
def grade_documents_tool(query: str, retrieved, top_k: int = 3) -> dict:
    # 1) 解析 retrieved
    def _coerce_to_dict(obj):
        # 已是 dict：直接回傳
        if isinstance(obj, dict):
            return obj
        # 期望是 JSON 字串
        if not isinstance(obj, str):
            return {}
        s = obj.strip()

        # 1) 先嘗試原生 loads
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) 用「括號平衡」從第一個 '{' 找到正好配平的結尾
        start = s.find('{')
        if start == -1:
            return {}

        depth = 0
        end = -1
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end != -1:
            candidate = s[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 3) 退而求其次：再試一次「到最後一個 '}]' 的位置」的切法
        try:
            last_close = s.rfind('}]')
            if last_close != -1:
                candidate = s[start:last_close+2]  # 包含 '}]'
                return json.loads(candidate)
        except Exception:
            pass

        # 4) 都不行就回空 dict
        return {}


    retrieved_dict = _coerce_to_dict(retrieved)
    docs_in = retrieved_dict.get("docs", []) if isinstance(retrieved_dict, dict) else []
    if not docs_in:
        return {"scored_docs": [], "expanded_docs": [], "context": "", "warning": "retrieved 空或缺少 docs"}

    # 2) 用 BGE Reranker（cross-encoder）重排
    #    說明：輸入 (query, doc) 配對，輸出每篇 doc 的相關性分數（已可直接排序）
    pairs = [[query, d["text"]] for d in docs_in]
    try:
        # normalize=True 會把分數做範圍正規化，排序更穩定
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as e:
        return {"scored_docs": [], "expanded_docs": [], "context": "", "error": f"reranker 失敗：{e}"}

    scored = []
    for d, s in zip(docs_in, scores):
        e = dict(d)
        e["score"] = float(s)
        scored.append(e)

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:max(1, min(top_k, len(scored)))]

    print("\n========== [Reranked Top Docs - BGE v2-m3] ==========")
    for i, d in enumerate(top):
        cid = d.get('meta', {}).get('CHUNK_ID')
        print(f"[{i}] score={d['score']:.3f} CHUNK_ID={cid}\n{d['text']}\n")

    # 3) 以 top_k 重建 Document，做 expand
    prod_id = extract_prod_id_from_query(query)
    vectorstore = build_vectorstore(prod_id) if prod_id else None
    base_docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in top]
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, base_docs) if vectorstore else base_docs

    # 4) 打包 expand 後的 docs 與 context
    packed = pack_docs(expanded_docs, max_docs=8)
    print("\n========== [Expanded Docs After Rerank] ==========")
    for i, d in enumerate(packed["docs"]):
        cid = d.get('meta', {}).get('CHUNK_ID')
        print(f"[{i}] CHUNK_ID={cid}\n{d['text']}\n")

    return {
        "scored_docs": top,
        "expanded_docs": packed["docs"],
        "context": packed["context"],
    }


def generate_answer_tool(query: str, context: str) -> dict:
    """
    只負責依 context 生成答案（不做檢索與重排）
    回傳：{"answer": "...", "context": context}
    """
    ctx = (context or "").strip()
    if not ctx:
        return {"answer": "找不到相關內容", "context": ""}

    chain = prompt | client
    result_msg = chain.invoke({"context": ctx, "query": query})
    answer = result_msg.content.strip()
    return {"answer": answer, "context": ctx}