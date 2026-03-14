from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import AzureChatOpenAI
from langchain.tools import StructuredTool
from functools import partial
from utils import extract_prod_id_from_query
from tools import (
    evaluate_answer_metrics,
    query_expanding_metrics,
    information_need_tool,   
    semantic_retriever_tool,
    keyword_retriever_tool,
    grade_documents_tool,
    generate_answer_tool,
)

from difficulty import predict_difficulty

from pydantic import BaseModel, Field
from prompt import *
from typing import List, Optional
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

current_product_name = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======== 設定 LLM ========
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0
)

# ======== 註冊 Tool ========
class InformationNeedInput(BaseModel):
    query: str = Field(description="使用者提出的保險問題（將被拆解為回答所需的知識資訊點清單）")

class SemanticRetrieverInput(BaseModel):
    query: str = Field(description="使用者原始問題")
    k: int = Field(default=8, description="最多回傳前 k 篇初步命中文件")
    threshold: float = Field(default=0.3, description="語意檢索最低分數門檻")

class KeywordRetrieverInput(BaseModel):
    query: str = Field(description="使用者原始問題")
    keywords: List[str] = Field(description="關鍵字清單，例如 ['失智症', '身故']")
    k: int = Field(default=8, description="最多回傳前 k 篇初步命中文件")

class GradeDocumentsInput(BaseModel):
    query: str = Field(description="使用者原始問題")
    retrieved: Optional[str | dict] = Field(
        description="來自任一 retriever 的輸出（必含 'docs'）。可為 JSON 字串或物件。"
    )
    top_k: int = Field(default=3, description="rerank 後保留的文件數量")

class GenerateAnswerInput(BaseModel):
    query: str = Field(description="使用者原始問題")
    context: str = Field(description="要餵給 LLM 的最終上下文（通常是 grade 後 expand 的結果）")

class EvaluationInput(BaseModel):
    query: str = Field(description="使用者提出的保險問題")
    information_points: str = Field(
        description="由 InformationNeedTool 產生的知識資訊點清單 (JSON 字串或陣列)"
    )
    answer: str = Field(description="模型產生的回覆")

class QueryExpansionInput(BaseModel):
    query: str = Field(description="使用者的原始問題")
    context: str = Field(description="檢索到的上下文內容")
    answer: str = Field(description="根據上下文產生的模型回覆")

information_need_agent_tool = StructuredTool.from_function(
    name="InformationNeedTool",
    func=information_need_tool,  # information_need_tool(query: str) -> dict
    description=(
        "將使用者問題拆解為『回答所需的知識資訊點』清單。"
        "回傳 JSON：{'info_needs': [{'id','description','must_have'} ...]}。"
    ),
    args_schema=InformationNeedInput
)

semantic_retriever_agent_tool = StructuredTool.from_function(
    name="SemanticRetrieverTool",
    func=semantic_retriever_tool,
    description="語意檢索：只做初步檢索（top-k，不 expand、不生成），回傳 {'docs': [...], 'context': '...'}。",
    args_schema=SemanticRetrieverInput
)

keyword_retriever_agent_tool = StructuredTool.from_function(
    name="KeywordBasedRetrieverTool",
    func=keyword_retriever_tool,
    description="關鍵字檢索：只做初步檢索（top-k，不 expand、不生成），回傳 {'docs': [...], 'context': '...'}。",
    args_schema=KeywordRetrieverInput
)

grade_documents_agent_tool = StructuredTool.from_function(
    name="GradeDocumentsTool",
    func=grade_documents_tool,
    description="對初步檢索結果做 rerank（cosine），取 top_k，再 expand 成最終 context。",
    args_schema=GradeDocumentsInput
)

generate_answer_agent_tool = StructuredTool.from_function(
    name="GenerateAnswerTool",
    func=generate_answer_tool,
    description="僅依已整理好的 context 生成答案（不做檢索與重排）。",
    args_schema=GenerateAnswerInput
)

answer_evaluator_tool = StructuredTool.from_function(
    name="AnswerEvaluatorTool",
    func=evaluate_answer_metrics,
    description=(
        "針對保險問題的回答進行評估："
        "根據 InformationNeedTool 提供的知識資訊點，"
        "並針對每個資訊點給出 0～1 分的評分。"
    ),
    args_schema=EvaluationInput
)

query_expanding_tool = StructuredTool.from_function(
    name="QueryExpandingTool",
    func=query_expanding_metrics,
    description="當無法有效回答時，根據原始保險問題改寫成更具檢索效率的新問題或關鍵字組",
    args_schema=QueryExpansionInput
)

tools = [
    information_need_agent_tool,   
    semantic_retriever_agent_tool,
    keyword_retriever_agent_tool,
    grade_documents_agent_tool,
    generate_answer_agent_tool,
    answer_evaluator_tool,
    query_expanding_tool,
]

# ======== 初始化 Agent ========
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# ======== CLI with reAct 回圈 ========
def should_react(eval_result: dict) -> bool:
    keys_to_check = ["Context Relevancy", "Context Adherence", "Answer Relevancy"]
    return any(eval_result.get(k, 1.0) <= 0.8 for k in keys_to_check)

def parse_scores_from_output(output: str) -> dict:
    try:
        # 嘗試找出最後一個 JSON 區塊
        json_match = re.findall(r"\{[\s\S]*?\}", output)
        if json_match:
            last_json_str = json_match[-1]
            return json.loads(last_json_str)
    except Exception as e:
        print("⚠️ JSON 解析失敗：", e)
    return {}


# ======== 互動式 CLI 模式 ========
while True:
    query = input("📝 請輸入保險問題（Enter 離開）：").strip()
    if not query:
        print("👋 結束保險問答")
        break

    current_product_name = extract_prod_id_from_query(query)
    print(f"📌 已鎖定產品名稱：{current_product_name}")

    difficulty = predict_difficulty(query)   # "easy" / "hard"
    print(f"🧠 難易度判斷：{difficulty}")


    print("🔍 正在分析並回答問題...")

    # ✅ easy：不進擴檢 reAct（只做一次檢索→回答→評估）
    if difficulty == "easy":
        wrapped_input = f"""
        你是一位保險問答助手，請嚴格依序完成以下流程，不能跳過：

        【步驟0｜產出資訊點】
        - 呼叫 `InformationNeedTool`，把使用者問題拆成「知識資訊點」清單（繁體中文）。
        - 核心要點標記 must_have=true，可補充少量 must_have=false。

        【步驟1｜檢索→重排→生成（僅一輪，不做擴檢）】
        - 先用 `SemanticRetrieverTool` 對原始問題做初步語意檢索，取得一個 JSON 物件。
        - 將檢索結果交給 `GradeDocumentsTool`，務必把 **SemanticRetrieverTool的原始輸出物件** 作為參數 `retrieved` 傳入，進行 rerank 並 expand，得到最終 context。
        - 使用 `GenerateAnswerTool` 依該 context 生成「答案 + 條文依據」。

        【步驟2｜評估（只評分，不觸發擴檢）】
        - 呼叫 `AnswerEvaluatorTool`，以【步驟0】的資訊點逐條打分（0～1）。
        - must_have=true 若 < 0.8，只需在評分結果中標註，不需要擴檢。

        【使用者問題】：
        {query}
        """
    else:
        # ✅ hard：保留你原本完整流程（最多3輪擴檢）
        wrapped_input = f"""
        你是一位保險問答助手，請嚴格依序完成以下流程，不能跳過：

        【步驟0｜產出資訊點】
        - 呼叫 `InformationNeedTool`，把使用者問題拆成「知識資訊點」清單（繁體中文）。
        - 核心要點標記 must_have=true，可補充少量 must_have=false。

        【步驟1｜第一次檢索→重排→生成】
        - 先用 `SemanticRetrieverTool` 對原始問題做初步語意檢索，取得一個 JSON 物件。
        - 將檢索結果交給 `GradeDocumentsTool`，務必把 **SemanticRetrieverTool的原始輸出物件** 作為參數 `retrieved` 傳入，進行 rerank 並 expand，得到最終 context。
        - 使用 `GenerateAnswerTool` 依該 context 生成「答案 + 條文依據」。

        【步驟2｜評估】
        - 呼叫 `AnswerEvaluatorTool`，以【步驟0】的資訊點逐條打分（0～1）。
        - 僅對 must_have=true 的點設門檻：分數需 ≥ 0.8；must_have=false 不設門檻。

        【步驟3｜不足則擴檢（最多3輪）】
        - 若任一 must_have=true 的分數 < 0.8：
        1) 呼叫 `QueryExpandingTool`，輸入「原始問題 + 當前 context + 目前答案」，取得：
            - `【精簡後的有用上下文】`（需作為 extra_context）
            - 下一步建議的類型與內容：
            - 若為「方式：semantic」：請再次使用 `SemanticRetrieverTool`，並帶入 extra_context；接著再走 `GradeDocumentsTool` → `GenerateAnswerTool`。
            - 若為「方式：keyword」：請使用 `KeywordBasedRetrieverTool`，並帶入 extra_context；接著再走 `GradeDocumentsTool` → `GenerateAnswerTool`。
        2) 產生新答案後，回到【步驟2】再次評估；最多重複三輪修正。

        【使用者問題】：
        {query}
        """
    result = agent.invoke({"input": wrapped_input})





    