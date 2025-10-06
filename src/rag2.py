from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain.schema import BaseRetriever
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from pydantic import Field
from dotenv import load_dotenv
from typing import List, Set, Optional, Tuple
import numpy as np
import jieba
import json
import os
import re

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("OPENAI_API_VERSION")

client = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    openai_api_version=api_version,
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        'trust_remote_code': True,
        'device': 'mps'  # 或改成 'cpu', 'cuda' 看你的電腦
    },
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8
    }
)

# 產品名稱、ID
product_mapping = {
    "國泰人壽真漾心安住院醫療終身保險": "AGG",
    "國泰人壽真康順手術醫療終身保險(外溢型)": "L66",
    "國泰人壽樂平安傷害保險": "GI1",
}

def extract_prod_id_from_query(query: str) -> Optional[str]:
    for prod_name, prod_id in product_mapping.items():
        if prod_name in query:
            return prod_id
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
            result.append((aid, title))  # 不再濾重 title
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
        expanded_docs.append(doc)

        if isinstance(chunk_id, int):
            seen_chunks.add(chunk_id)
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

    # debug print
    print("\n📊 向量語意分數（appendix chunks）：")
    for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True):
        cid = doc.metadata.get("CHUNK_ID")
        title = doc.metadata.get("TITLE")
        print(f"- CHUNK_ID: {cid}, TITLE: {title}, Score: {score:.4f}")

    return [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]]

# Prompt 設定
decision_prompt_template = """
###Instruction###
You are a professional assistant specializing in Taiwanese insurance policies. Based on the following insurance clause content and appendix titles, please determine whether the provided information is sufficient to answer the user's question.

###Task###
- Based on the following insurance clause content and appendix titles, determine whether you can directly answer the user's question.


- If you can answer the question with the provided context, please provide a full response using the format below:
- The section below may contain 2 to 5 different insurance clause snippets. Identify and extract only the relevant information needed to answer the user's question. Your answer must be based strictly on the information in the provided context. Avoid introducing content that is not explicitly mentioned. If the answer cannot be found, say:「找不到相關內容」.

回答：
[Your answer in Traditional Chinese. Do not repeat the user's question.]

條文依據：
[Cite relevant clause or article if applicable. If none, say:「找不到相關內容」.]

###Reference Example - DO NOT COPY, FOR STYLE ONLY###
Example Question:  
國泰人壽真漾心安住院醫療終身保險保障範圍包括哪些項目？

Example Answer:  
回答：  
保障項目包含：住院醫療保險金、加護病房或燒燙傷病房保險金、祝壽保險金、身故保險金或喪葬費用保險金，以及所繳保險費的退還。  
條文依據：  
摘要中明確列出上述保障項目，為本商品之主要給付項目。


- If you cannot answer the question without more information, please reply in this format.
- You must extract `needed_appendix_titles` only from the provided Appendix Titles list.
{{ "answerable": false, "needed_appendix_titles": [""] }}


###User Question###
{question}

###Context###
The following insurance clause content was retrieved based on the user's question:

{context}

###Appendix Titles (Potentially Related)###
{appendix_titles}
"""
decision_prompt = PromptTemplate(
    template=decision_prompt_template,
    input_variables=["question", "context", "appendix_titles"]
)

prompt_template = """
###Instruction###
###Instruction###
You are a professional assistant specializing in Taiwanese insurance policies.

###Task###
Your job is to help users understand specific insurance clauses based on the provided context. Focus on delivering accurate, concise answers in Traditional Chinese.

###Context###
The section below may contain 2 to 5 different insurance clause snippets. Identify and extract only the relevant information needed to answer the user's question. Your answer must be based strictly on the information in the provided context. Avoid introducing content that is not explicitly mentioned. If the answer cannot be found, say:「找不到相關內容」.

###Format###
Please begin your response with「回答：」and follow the exact format below:

回答：
[Your answer in Traditional Chinese. Do not repeat the user's question.]

條文依據：
[Cite relevant clause or article if applicable. If none, say:「找不到相關內容」.]

###Reference Example - DO NOT COPY, FOR STYLE ONLY###
Example Question:  
國泰人壽真漾心安住院醫療終身保險保障範圍包括哪些項目？

Example Answer:  
回答：  
保障項目包含：住院醫療保險金、加護病房或燒燙傷病房保險金、祝壽保險金、身故保險金或喪葬費用保險金，以及所繳保險費的退還。  
條文依據：  
摘要中明確列出上述保障項目，為本商品之主要給付項目。

---

###保險條款資料：
{context}

###使用者問題：
{question}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 靜態 retriever 類別
class StaticRetriever(BaseRetriever):
    docs: List[Document]
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

class FilteredRetriever(BaseRetriever):
    vectorstore: Chroma = Field(...)
    threshold: float = Field(default=0.3)
    k: int = Field(default=2)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_and_scores: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k
        )
        return [doc for doc, score in docs_and_scores if score >= self.threshold]

# 建立 QA chain
def build_retrieval_qa_chain(client, docs, prompt_template=prompt):
    if len(docs) > 5:
        docs = docs[:5]
    retriever = StaticRetriever(docs=docs)
    return RetrievalQA.from_chain_type(
        llm=client,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

# stage 1
def should_answer_directly(client, question: str, docs: List[Document], appendix_titles: List[Tuple[int, str]]) -> str:
    appendix_title_str = "\n".join([f"{aid}: {title}" for aid, title in appendix_titles])
    context_str = "\n\n".join([doc.page_content for doc in docs])
    print("\nappendix_titles 提供給 LLM：")
    print(appendix_title_str)
    
    # 使用原始 LLMChain 處理 decision_prompt
    chain = LLMChain(llm=client, prompt=decision_prompt)
    return chain.invoke({
        "question": question,
        "context": context_str,
        "appendix_titles": appendix_title_str
    })["text"].strip()


while True:
    query = input("👉 請輸入保險問題（Enter 離開）：").strip()
    if not query:
        print("👋 已離開保險問答系統")
        break

    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        print("無法判斷產品名稱，請確認輸入是否包含有效保單名稱。")
        continue

    # 建立向量資料庫與基本檢索
    vectorstore = Chroma(persist_directory=f"../chroma_db/{prod_id}", embedding_function=embeddings)
    softlink_map = load_softlink_mapping(prod_id)
    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=0.3, k=2)
    retrieved_docs = retriever.invoke(query)
    expanded_docs = expand_retrieved_chunks_v2(vectorstore, retrieved_docs)
    if len(expanded_docs) > 5:
        expanded_docs = expanded_docs[:5]

    # 找出有關聯的 appendix titles
    retrieved_chunk_ids = {doc.metadata.get("CHUNK_ID") for doc in retrieved_docs if isinstance(doc.metadata.get("CHUNK_ID"), int)}
    appendix_list = get_softlinked_appendix_titles(vectorstore, softlink_map, retrieved_chunk_ids)
    appendix_title_str = "\n".join([f"{aid}: {title}" for aid, title in appendix_list])

    # 執行第一階段判斷
    decision_output = should_answer_directly(client, query, expanded_docs, appendix_list)



    # 否則解析 JSON 回傳的 needed_appendix_titles
    useful_chunk_titles = []
    needed_titles = []
    if '"answerable": false' in decision_output:
        try:
            json_start = decision_output.rfind("{")
            json_str = decision_output[json_start:]
            parsed = json.loads(json_str)
            needed_titles = parsed.get("needed_appendix_titles", [])

            print(f"🔍 needed_appendix_titles from LLM: {needed_titles}")

            appendix_docs = []
            
            seen_ids = {doc.metadata.get("CHUNK_ID") for doc in expanded_docs}

            # 收集所有符合 title 的 appendix chunk（最多取 1 筆）
            for needed_title in needed_titles:
                matching_docs = []

                for aid, title in appendix_list:
                    if title == needed_title:
                        docs = vectorstore.get(where={"CHUNK_ID": aid}, include=["documents", "metadatas"])
                        for content, meta in zip(docs["documents"], docs["metadatas"]):
                            cid = meta.get("CHUNK_ID")
                            if cid not in seen_ids:
                                matching_docs.append(Document(page_content=content, metadata=meta))

                # 從中選出最 relevant 的 1 筆
                top_docs = rerank_appendix_with_embedding(query, matching_docs, embeddings, top_k=1)
                appendix_docs.extend(top_docs)
                seen_ids.update(doc.metadata.get("CHUNK_ID") for doc in top_docs)
                            
            print(f"\n📎 appendix_docs 抓到 {len(appendix_docs)} 筆：")
            for doc in appendix_docs:
                print(f"- CHUNK_ID: {doc.metadata.get('CHUNK_ID')}, TITLE: {doc.metadata.get('TITLE')}")

            # 限制各自最多 5 筆
            limited_expanded = expanded_docs[:5]
            limited_appendix = appendix_docs[:2]
            final_docs = limited_expanded + limited_appendix
            rag_chain = build_retrieval_qa_chain(client, final_docs, prompt)
            response = rag_chain.invoke({"query": query})

            print("\n✅ 第二階段回答：")
            print(response["result"])
            print("-" * 60)
            print("\n引用來源文件：")
            for i, doc in enumerate(response["source_documents"]):
                print(f"\n[{i+1}] 來自：{doc.metadata}")
                print(doc.page_content, "\n")

        except Exception as e:
            print("⚠ 第二階段處理失敗：", e)

    else:
        print(decision_output)
        print("-" * 60)
        print("\n引用來源文件：")
        for i, doc in enumerate(expanded_docs):
            print(f"\n[{i+1}] 來自：{doc.metadata}")
            print(doc.page_content, "\n")
