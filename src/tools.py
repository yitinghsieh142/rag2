from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable
from langchain.schema import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import json
from dotenv import load_dotenv
from typing import List, Set, Tuple, Optional

from utils import (
    extract_prod_id_from_query,
    build_vectorstore,
    load_softlink_mapping,
    get_softlinked_appendix_titles,
    expand_retrieved_chunks_v2,
    rerank_appendix_with_embedding,
    build_retrieval_qa_chain,
    FilteredRetriever,
    keyword_based_retriever,
)

from prompt import decision_prompt, prompt, answer_evaluation_prompt, query_expanding_prompt

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

# === tool #1：使用初步 retrieve 資訊回答問題，或判斷是否使用額外appendix ===
def stage1_decision_tool(query: str, extra_context: Optional[str] = None, default_product_name: Optional[str] = None) -> str:
    # 如果 query 中沒有產品名稱，嘗試補上
    prod_id = extract_prod_id_from_query(query)
    print(default_product_name)
    print(prod_id)
    if not prod_id and default_product_name:
        query = f"{default_product_name} {query}"
        prod_id = extract_prod_id_from_query(query)
        print("hi")
    elif prod_id:
        current_product_name = prod_id

    vectorstore = build_vectorstore(prod_id)
    softlink_map = load_softlink_mapping(prod_id)

    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=0.3, k=2)
    print(query)
    retrieved_docs = retriever.invoke(query)
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, retrieved_docs)
    if len(expanded_docs) > 5:
        expanded_docs = expanded_docs[:5]

    print("\n========== [Stage1] Retrieved Docs ==========")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    print("\n========== [Stage1] Expanded Docs ==========")
    for i, doc in enumerate(expanded_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")


    retrieved_chunk_ids = {doc.metadata.get("CHUNK_ID") for doc in retrieved_docs if isinstance(doc.metadata.get("CHUNK_ID"), int)}
    appendix_list = get_softlinked_appendix_titles(vectorstore, softlink_map, retrieved_chunk_ids)
    appendix_title_str = "\n".join([f"{aid}: {title}" for aid, title in appendix_list])
    context_str = "\n\n".join([doc.page_content for doc in expanded_docs])

    if extra_context:
        context_str = f"{context_str}\n---\n{extra_context}"
        
    decision_chain: Runnable = decision_prompt | client
    result_msg = decision_chain.invoke({
        "question": query,
        "context": context_str,
        "appendix_titles": appendix_title_str
    })
    answer = result_msg.content.strip()

    return {
        "answer": answer,
        "context": context_str,
    }

# === tool #2：使用擴增的appendix title 回答問題 ===
def stage2_rag_with_appendix_tool(query: str, needed_titles: list[str]) -> str:
    if not query or not isinstance(needed_titles, list):
        return "❌ 請確認輸入格式：{\"query\": ..., \"needed_titles\": [...]}"

    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        return "❌ 無法判斷產品名稱"

    vectorstore = build_vectorstore(prod_id)
    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=0.3, k=2)
    retrieved_docs = retriever.invoke(query)
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, retrieved_docs)
    if len(expanded_docs) > 5:
        expanded_docs = expanded_docs[:5]

    softlink_map = load_softlink_mapping(prod_id)
    retrieved_chunk_ids = {
        doc.metadata.get("CHUNK_ID") for doc in retrieved_docs if isinstance(doc.metadata.get("CHUNK_ID"), int)
    }
    appendix_list = get_softlinked_appendix_titles(vectorstore, softlink_map, retrieved_chunk_ids)

    appendix_docs = []
    seen_ids = {doc.metadata.get("CHUNK_ID") for doc in expanded_docs}

    for needed_title in needed_titles:
        # 去除前綴數字與冒號，例如 "38: 附表：增額係數表" -> "附表：增額係數表"
        needed_title = needed_title.split(":", 1)[-1].strip()

        matching_docs = []
        for aid, title in appendix_list:
            if title == needed_title:
                docs = vectorstore.get(where={"CHUNK_ID": aid}, include=["documents", "metadatas"])
                for content, meta in zip(docs["documents"], docs["metadatas"]):
                    cid = meta.get("CHUNK_ID")
                    if cid not in seen_ids:
                        matching_docs.append(Document(page_content=content, metadata=meta))
        top_docs = rerank_appendix_with_embedding(query, matching_docs, embeddings, top_k=1)
        appendix_docs.extend(top_docs)

    final_docs = expanded_docs[:5] + appendix_docs[:2]
    qa_chain = build_retrieval_qa_chain(client, final_docs, prompt)
    # context_str = "\n\n".join([doc.page_content for doc in final_docs])
    print("\n========== [Stage2] Retrieved Docs ==========")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    print("\n========== [Stage2] Expanded Docs ==========")
    for i, doc in enumerate(expanded_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    print("\n========== [Stage2] Appendix Docs ==========")
    for i, doc in enumerate(appendix_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    print(query)
    try:
        # result = qa_chain.invoke({
        #     "query": query,
        #     # "context": context_str
        # })
        result = qa_chain({ "query": query })
        return result
    except Exception as e:
        return f"❌ 回答失敗哈哈：{e}"
    
def stage2_rag_with_appendix_tool_wrapper(query: str, needed_titles: list[str]) -> str:
    print("[wrapper] query:", query)
    print("[wrapper] needed_titles:", needed_titles)
    return stage2_rag_with_appendix_tool(query, needed_titles)

# === tool #3：各指標答案評分 ===
def evaluate_answer_metrics(query: str, context: str, answer: str) -> str:
    chain = answer_evaluation_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "context": context,
        "answer": answer
    })
    return result_msg.content.strip()

# === tool #4：修改 query ===
def query_expanding_metrics(query: str, context: str, answer: str) -> str:
    chain = query_expanding_prompt | client
    result_msg = chain.invoke({
        "query": query,
        "context": context,
        "answer": answer
    })
    return result_msg.content.strip()

# === tool #5：Keyword-based Retriever Tool ===
def keyword_retriever_tool(query: str, keywords: List[str], extra_context: Optional[str] = None) -> dict:
    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        return "❌ 無法判斷產品名稱"

    vectorstore = build_vectorstore(prod_id)

    matched_docs = keyword_based_retriever(vectorstore, query, keywords)

    if not matched_docs:
        return "⚠️ 關鍵字檢索未命中任何條文內容"
    
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, matched_docs)
    if len(expanded_docs) > 5:
        expanded_docs = expanded_docs[:5]


    print("\n========== [Keyword-based Retrieved Docs] ==========")
    for i, doc in enumerate(matched_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    print("\n========== [Stage1] Expanded Docs ==========")
    for i, doc in enumerate(expanded_docs):
        print(f"[{i}] CHUNK_ID={doc.metadata.get('CHUNK_ID')}\n{doc.page_content}\n")

    keyword_context = "\n---\n".join([doc.page_content for doc in expanded_docs[:5]])
    combined_context = f"{extra_context}\n---\n{keyword_context}" if extra_context else keyword_context

    # 使用同樣 QA prompt 回答
    # qa_chain = build_retrieval_qa_chain(client, matched_docs[:3], prompt)
    # result = qa_chain.invoke({"query": query})
    # answer = result["result"] if isinstance(result, dict) else result
    qa_chain = prompt | client
    result_msg = qa_chain.invoke({
        "context": combined_context,
        "query": query
    })
    answer = result_msg.content.strip()

    return {
        "answer": answer,
        "context": combined_context,
    }