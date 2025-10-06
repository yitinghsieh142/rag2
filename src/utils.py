import os
import json
import numpy as np
from typing import List, Set, Tuple, Optional
from langchain.schema import Document, BaseRetriever
from pydantic import Field
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda
import jieba
from rank_bm25 import BM25Okapi

from langchain_community.embeddings import HuggingFaceEmbeddings

# === 向量儲存 ===
def build_vectorstore(prod_id: str):
    return Chroma(persist_directory=f"../chroma_db/{prod_id}", embedding_function=embeddings)

# === embeedding 模型 ===
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'trust_remote_code': True, 'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
)

# === 產品名稱對應表 ===
product_mapping = {
    "國泰人壽真漾心安住院醫療終身保險": "AGG",
    "國泰人壽真康順手術醫療終身保險(外溢型)": "L66",
    "國泰人壽樂平安傷害保險": "GI1",
    "自由配重大疾病(甲型)":"CFG",
    "自由配重大傷病":"CFE",
    "新鍾心滿福":"ZCN",
    "滿溢寶":"FSAFSB",
    "樂鍾心":"SC1",
    "鍾心滿溢":"ZDN",
    "鍾溢祝福":"PX5",
    "醫心康愛":"DPA"
}

product_name_to_id = product_mapping
product_id_to_id = {v: v for v in product_mapping.values()}
full_product_lookup = {**product_name_to_id, **product_id_to_id}
def extract_prod_id_from_query(query: str) -> Optional[str]:
    for key in full_product_lookup:
        if key in query:
            return full_product_lookup[key]
    return None

# === 載入 softlink 映射表 ===
def load_softlink_mapping(prod_id: str) -> dict:
    path = f"../soft_links_output/{prod_id}.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === 擴展 retrieved sections 對應的 appendix ===
def get_softlinked_appendix_titles(vectorstore, softlink_map: dict, retrieved_chunk_ids: Set[int]) -> List[Tuple[int, str]]:
    appendix_ids = [int(aid) for aid, sections in softlink_map.items() if any(sec in retrieved_chunk_ids for sec in sections)]
    result = []
    for aid in appendix_ids:
        docs = vectorstore.get(where={"CHUNK_ID": aid}, include=["documents", "metadatas"])
        for content, meta in zip(docs["documents"], docs["metadatas"]):
            title = meta.get("TITLE", "")
            result.append((aid, title))
    return result

# === group / appendix 擴展 ===
def expand_retrieved_chunks_v2(vectorstore, retrieved_docs: List[Document]) -> List[Document]:
    group_set: Set[str] = set()
    appendix_chunk_ids: Set[int] = set()
    seen_chunks: Set[int] = set()
    expanded_docs: List[Document] = []

    # Step 1: collect group + related_appendix
    for doc in retrieved_docs:
        chunk_id = doc.metadata.get("CHUNK_ID")
        group = doc.metadata.get("GROUP")
        related_appendix = doc.metadata.get("RELATED_APPENDIX")

        # 判斷是否為 appendix：group 開頭是 "appendix-"
        is_appendix = isinstance(group, str) and group.startswith("appendix-")

        expanded_docs.append(doc)

        if isinstance(chunk_id, int):
            seen_chunks.add(chunk_id)
        if not is_appendix:
            # 只有 section 才擴展
            if isinstance(group, str):
                group_set.add(group)
                print("hihi")
            if isinstance(related_appendix, int):
                appendix_chunk_ids.add(related_appendix)
    
    # Step 2: 查詢相同 group 的 chunk
    for group in group_set:
        group_docs = vectorstore.get(where={"GROUP": group}, include=["documents", "metadatas"])
        for content, meta in zip(group_docs["documents"], group_docs["metadatas"]):
            cid = meta.get("CHUNK_ID")
            if isinstance(cid, int) and cid not in seen_chunks:
                expanded_docs.append(Document(page_content=content, metadata=meta))
                seen_chunks.add(cid)
    # Step 3: 查詢 related appendix chunk
    for appendix_id in appendix_chunk_ids:
        appendix_docs = vectorstore.get(where={"CHUNK_ID": appendix_id}, include=["documents", "metadatas"])
        for content, meta in zip(appendix_docs["documents"], appendix_docs["metadatas"]):
            cid = meta.get("CHUNK_ID")
            if isinstance(cid, int) and cid not in seen_chunks:
                expanded_docs.append(Document(page_content=content, metadata=meta))
                seen_chunks.add(cid)

    return expanded_docs

# === reranker ===
def rerank_appendix_with_embedding(query: str, appendix_docs: List[Document], embedding_model, top_k=3) -> List[Document]:
    if not appendix_docs:
        return []

    query_embedding = embedding_model.embed_query(query)
    doc_texts = [doc.page_content for doc in appendix_docs]
    doc_embeddings = embedding_model.embed_documents(doc_texts)

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_docs = [
        (doc, cosine_similarity(query_embedding, doc_emb))
        for doc, doc_emb in zip(appendix_docs, doc_embeddings)
    ]

    return [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]]

# === RetrievalQA 使用的靜態 retriever 類別 ===
class StaticRetriever(BaseRetriever):
    docs: List[Document]
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

class FilteredRetriever(BaseRetriever):
    vectorstore: Chroma = Field(...)
    threshold: float = Field(default=0.3)
    k: int = Field(default=2)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        return [doc for doc, score in docs_and_scores if score >= self.threshold]

# === 組建 RetrievalQA Chain ===

def build_retrieval_qa_chain(client, docs: List[Document], prompt_template):
    if len(docs) > 5:
        docs = docs[:5]

    context_str = "\n\n".join([doc.page_content for doc in docs])
    chain = LLMChain(llm=client, prompt=prompt_template)

    def invoke_fn(inputs: dict):
        query = inputs["query"]
        return chain.invoke({
            "context": context_str,
            "query": query
        })

    return RunnableLambda(invoke_fn) # 回傳 callable

# === Keyword-based retriever ===
# def keyword_based_retriever(vectorstore: Chroma, keywords: List[str], top_k: int = 3) -> List[Document]:
#     matched_docs = []
#     seen_chunk_ids = set()

#     # 取得所有 chunks（使用 include 取得 metadata）
#     all_chunks = vectorstore.get(include=["documents", "metadatas"])

#     for content, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
#         if any(kw in content for kw in keywords):
#             cid = meta.get("CHUNK_ID")
#             if cid not in seen_chunk_ids:
#                 matched_docs.append(Document(page_content=content, metadata=meta))
#                 seen_chunk_ids.add(cid)
#                 if len(matched_docs) >= top_k:
#                     break  # 最多 top_k 筆

#     return matched_docs

# def keyword_based_retriever(vectorstore: Chroma, query: str, keywords: List[str], top_k: int = 3) -> List[Document]:
#     matched_docs = []
#     seen_chunk_ids = set()

#     # 取得所有 chunks（使用 include 取得 metadata）
#     all_chunks = vectorstore.get(include=["documents", "metadatas"])

#     # 先找出最多 10 筆 keyword 命中資料
#     for content, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
#         if any(kw in content for kw in keywords):
#             cid = meta.get("CHUNK_ID")
#             if cid not in seen_chunk_ids:
#                 matched_docs.append(Document(page_content=content, metadata=meta))
#                 seen_chunk_ids.add(cid)
#                 if len(matched_docs) >= 10:
#                     break  # 最多先取 10 筆

#     if not matched_docs:
#         return []

#     # 計算語意相似度 rerank
#     query_embedding = embeddings.embed_query(query)
#     doc_embeddings = embeddings.embed_documents([doc.page_content for doc in matched_docs])

#     def cosine_similarity(a, b):
#         a = np.array(a)
#         b = np.array(b)
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     scored_docs = [
#         (doc, cosine_similarity(query_embedding, emb))
#         for doc, emb in zip(matched_docs, doc_embeddings)
#     ]
#     top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]

#     return [doc for doc, _ in top_docs]

def keyword_based_retriever(vectorstore: Chroma, query: str, keywords: List[str], top_k: int = 3) -> List[Document]:
    seen_chunk_ids = set()
    all_chunks = vectorstore.get(include=["documents", "metadatas"])

    # 預處理文本與分詞
    corpus = []
    doc_list = []
    for content, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
        tokens = list(jieba.cut(content))
        corpus.append(tokens)
        doc_list.append(Document(page_content=content, metadata=meta))

    # 用 query 的關鍵字分詞當作查詢
    tokenized_query = keywords  # 若已分好詞，可直接用

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    # 排序後取前 k 筆，避免重複 chunk
    doc_scores = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)
    matched_docs = []
    for doc, _ in doc_scores:
        cid = doc.metadata.get("CHUNK_ID")
        if cid not in seen_chunk_ids:
            matched_docs.append(doc)
            seen_chunk_ids.add(cid)
            if len(matched_docs) >= top_k:
                break

    return matched_docs

