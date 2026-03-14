import os
import json
import numpy as np
import re
from typing import Dict, List, Set, Tuple, Optional
from langchain.schema import Document, BaseRetriever
from pydantic import Field
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda

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


# --- 共用小工具（打包 docs 與 context） ---
def pack_docs(docs: List[Document], max_docs: int = 8) -> dict:
    items = []
    for d in docs[:max_docs]:
        items.append({
            "doc_id": d.metadata.get("CHUNK_ID") or d.metadata.get("id") or d.metadata.get("doc_id"),
            "text": d.page_content,
            "meta": d.metadata,
        })
    context = "\n---\n".join([x["text"] for x in items])
    return {"docs": items, "context": context}

