from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json
from tqdm import tqdm
from uuid import uuid4

# 設定 Hugging Face 的免費 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        'trust_remote_code': True,
        'device': 'mps'
    },
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8 
    }
)

# 向量儲存處理
output_folder = "../output"
all_docs = []
# 跑全部
# prod_folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
# 跑特定檔案
prod_folders = ["CFG"]
prod_folders = [f for f in prod_folders if os.path.isdir(os.path.join(output_folder, f))]

for prod_id in tqdm(prod_folders, desc="Processing Products"):
    prod_folder_path = os.path.join(output_folder, prod_id)

    # 遍歷該產品資料夾內的 JSON 文件
    for json_file in os.listdir(prod_folder_path):
        if not json_file.endswith(".json"):
            continue  # 只處理 JSON 檔案
        
        json_path = os.path.join(prod_folder_path, json_file)
        
        # 讀取 JSON 檔案
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 取得產品名稱
        prod_name = data.get("metadata", {}).get("insurance_name", "Unknown")

        # 儲存要處理的 Document
        doc_list = []

        # 處理 metadata（合併內容）
        metadata_fields = ["payout_items", "contact_info", "summary", "other_metadata", "announcement_history"]
        # combined_metadata = {}

        for field in metadata_fields:
            value = data.get("metadata", {}).get(field, None)

            # 處理 `other_metadata` 和 `announcement_history`（合併 List）
            if isinstance(value, list):  
                # combined_metadata[field] = "；".join(value)  # 轉成單一字串
                value = "；".join(value)
            elif isinstance(value, str) and value.strip():  
                # combined_metadata[field] = value.strip()  # 去除空格
                value = value.strip()
            if value:
                doc_list.append(
                    Document(
                        page_content=value,
                        metadata={
                            "PROD_ID": prod_id,
                            "PROD_SNAME": prod_name,
                            "TYPE": "metadata",
                            "TITLE": field  # 新增 TITLE 欄位
                        }
                    )
                )

        for section in data.get("sections", []):
            title = section.get("title", "")
            content = section.get("content", "")
            chunk_id = section.get("chunk_id", None)
            group = section.get("group", "")
            related_appendix = section.get("related_appendix", None)

            metadata = {
                "PROD_ID": prod_id,
                "PROD_SNAME": prod_name,
                "TYPE": "sections",
                "TITLE": title,
                "CHUNK_ID": chunk_id,
                "GROUP": group,
            }
            if related_appendix is not None:
                metadata["RELATED_APPENDIX"] = related_appendix

            doc_list.append(
                Document(
                    page_content=f"{title}\n{content}",
                    metadata=metadata
                )
            )

        # 處理 appendices（附加條款）
        for appendix in data.get("appendices", []):
            title = appendix.get("title", "")
            content = appendix.get("content", "")
            chunk_id = appendix.get("chunk_id", None)
            group = appendix.get("group", "")

            doc_list.append(
                Document(
                    page_content=f"{title} - {content}",
                    # metadata={"PROD_ID": prod_id, "PROD_SNAME": prod_name, "TYPE": "appendices", "TITLE": title}
                    metadata={
                        "PROD_ID": prod_id,
                        "PROD_SNAME": prod_name,
                        "TYPE": "appendices",
                        "TITLE": title,
                        "CHUNK_ID": chunk_id,
                        "GROUP": group
                    }
                )
            )

    if doc_list:
        persist_path = f"../chroma_db/{prod_id}"
        vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        uuids = [str(uuid4()) for _ in range(len(doc_list))]
        vectorstore.add_documents(documents=doc_list, ids=uuids)


print("向量儲存完成！")
