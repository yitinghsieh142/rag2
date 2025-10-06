from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.schema import BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from pydantic import Field
from typing import List, Set, Optional
import json
import os

# 設定 LLM
AZURE_OPENAI_API_KEY = '101d45006bfb4efc9f98fbb9e258798d'
endpoint = os.getenv("ENDPOINT_URL", "https://cathay-translation-9h00100.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "9h00100-channel-intern-gpt-4o")
api_version = "2024-02-15-preview"

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
def expand_with_softlink_appendices(vectorstore, retrieved_docs: List[Document], softlink_map: dict) -> List[Document]:
    section_ids = set(doc.metadata.get("CHUNK_ID") for doc in retrieved_docs)
    seen_chunks = set(doc.metadata.get("CHUNK_ID") for doc in retrieved_docs)
    expanded_docs = list(retrieved_docs)

    for appendix_id_str, linked_sections in softlink_map.items():
        appendix_id = int(appendix_id_str)
        if any(sec_id in section_ids for sec_id in linked_sections):
            appendix_docs = vectorstore.get(where={"CHUNK_ID": appendix_id}, include=["documents", "metadatas"])
            for content, meta in zip(appendix_docs["documents"], appendix_docs["metadatas"]):
                cid = meta.get("CHUNK_ID")
                if cid not in seen_chunks:
                    expanded_docs.append(Document(page_content=content, metadata=meta))
                    seen_chunks.add(cid)

    return expanded_docs

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

# Prompt 設定
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
def build_retrieval_qa_chain(client, docs):
    if len(docs) > 5:
        docs = docs[:5]
    retriever = StaticRetriever(docs=docs)
    return RetrievalQA.from_chain_type(
        llm=client,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# while True:
#     query = input("👉 請輸入保險問題（Enter 離開）：").strip()
#     if not query:
#         print("👋 已離開保險問答系統")
#         break

#     prod_id = extract_prod_id_from_query(query)
#     if not prod_id:
#         print("無法判斷產品名稱，請確認輸入是否包含有效保單名稱。")
#         continue

#     vectorstore = Chroma(persist_directory=f"../chroma_db/{prod_id}", embedding_function=embeddings)
#     retriever = FilteredRetriever(vectorstore=vectorstore, threshold=0.3, k=2)
#     retrieved_docs = retriever.invoke(query)
#     expanded_docs = expand_retrieved_chunks_v2(vectorstore, retrieved_docs)
#     softlink_map = load_softlink_mapping(prod_id)
#     expanded_docs = expand_with_softlink_appendices(vectorstore, expanded_docs, softlink_map)

#     rag_chain = build_retrieval_qa_chain(client, expanded_docs)
#     response = rag_chain.invoke({"query": query})

#     print(response["result"])
#     print("-" * 60)
#     print("\n引用來源文件：")
#     for i, doc in enumerate(response["source_documents"]):
#         print(f"\n[{i+1}] 來自：{doc.metadata}")
#         print(doc.page_content, "\n")

while True:
    query = input("👉 請輸入保險問題（Enter 離開）：").strip()
    if not query:
        print("👋 已離開保險問答系統")
        break

    prod_id = extract_prod_id_from_query(query)
    if not prod_id:
        print("❌ 無法判斷產品名稱，請確認輸入是否包含有效保單名稱。")
        continue

    vectorstore = Chroma(persist_directory=f"../chroma_db/{prod_id}", embedding_function=embeddings)
    retriever = FilteredRetriever(vectorstore=vectorstore, threshold=0.3, k=2)
    retrieved_docs = retriever.invoke(query)

    print(f"\n🔍 初步擷取的 chunks：{len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[初步擷取 {i+1}] {doc.metadata}")
        print(doc.page_content)

    expanded_docs = expand_retrieved_chunks_v2(vectorstore, retrieved_docs)
    softlink_map = load_softlink_mapping(prod_id)
    expanded_docs = expand_with_softlink_appendices(vectorstore, expanded_docs, softlink_map)

    print(f"\n📎 擴展後總共 {len(expanded_docs)} 個 chunks（包含 group / related appendix / softlink）")
    for i, doc in enumerate(expanded_docs):
        print(f"\n[擴展 {i+1}] {doc.metadata}")
        print(doc.page_content)
        print("-" * 60)
