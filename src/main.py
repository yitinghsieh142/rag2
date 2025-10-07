from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import AzureChatOpenAI
from langchain.tools import StructuredTool
from functools import partial
from utils import extract_prod_id_from_query
from tools import (
    # stage1_decision_tool,
    evaluate_answer_metrics,
    query_expanding_metrics,
    information_need_tool,   
    semantic_retriever_tool,
    keyword_retriever_tool,
    grade_documents_tool,
    generate_answer_tool,
)
from pydantic import BaseModel, Field
from prompt import *
from typing import List, Optional
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

current_product_name = None
# def stage1_tool_wrapper(query: str, extra_context: Optional[str] = None):
#     return stage1_decision_tool(
#         query=query,
#         default_product_name=current_product_name,
#         extra_context=extra_context
#     )

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

# class KeywordRetrieverInput(BaseModel):
#     query: str = Field(description="使用者提出的保險問題")
#     keywords: List[str] = Field(description="關鍵字清單，例如 ['失智症', '身故']")
#     extra_context: Optional[str] = Field(default=None, description="從上次回答中留下的精簡上下文")


# class Stage1DecisionInput(BaseModel):
#     query: str = Field(description="使用者提出的保險問題")
#     extra_context: Optional[str] = Field(default=None, description="從上次回答中留下的精簡上下文")

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

# decision_tool = StructuredTool.from_function(
#     name="Stage1DecisionTool",
#     func=stage1_tool_wrapper,
#     description="使用初始 context（包含選擇性額外上下文 extra_context）回答保險問題，若有上輪留下的【精簡後有用上下文】，請務必放入 extra_context 欄位中",
#     args_schema=Stage1DecisionInput
# )

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

# keyword_tool = StructuredTool.from_function(
#     name="KeywordBasedRetrieverTool",
#     func=keyword_retriever_tool,
#     description="使用初始 context（包含選擇性額外上下文 extra_context）回答保險問題，若有上輪留下的【精簡後有用上下文】，請一定要放入 extra_context 欄位中",
#     args_schema=KeywordRetrieverInput
# )

tools = [
    information_need_agent_tool,   
    # decision_tool,
    semantic_retriever_agent_tool,
    keyword_retriever_agent_tool,
    grade_documents_agent_tool,
    generate_answer_agent_tool,
    answer_evaluator_tool,
    query_expanding_tool,
    # keyword_tool,
]

# ======== 初始化 Agent ========
agent = initialize_agent(
    tools=tools,
    llm=llm,
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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

    if query:
        current_product_name = extract_prod_id_from_query(query)
        print(f"📌 已鎖定產品名稱：{current_product_name}")

    print("🔍 正在分析並回答問題...")

    # wrapped_input = f"""
    # 你是一位保險問答助手，請依照以下流程一步步完成，不能跳過：

    # 步驟零：先使用 `InformationNeedTool`，把使用者問題拆解為「回答所需的知識資訊點」清單（以繁體中文撰寫）。此清單後續評分與檢索都要用到。
    # - 僅針對問題本身的核心需求；可補充少量 must_have=false 的輔助點。

    # 步驟一：使用檢索內容回答使用者的保險問題，產出「答案 + 條文依據」。

    # 步驟二：使用 `AnswerEvaluatorTool`，根據【步驟零】得到的資訊點清單，逐條評估答案覆蓋度，並針對每個資訊點給出 0～1 的分數。

    # 步驟三：若 **任何 must_have=true 的資訊點** 分數 **低於 0.8**，請使用 `QueryExpandingTool` 根據「原始問題 + 上下文 + 回答」找出缺乏的資訊，並產生新的語意問題或關鍵詞組進一步檢索。  
    # （注意：**must_have=false** 的分數不設門檻，不需觸發此步驟。）

    # 步驟四：請根據 `QueryExpandingTool` 產生的新問題或是關鍵詞組，執行進一步的檢索，並將上一步的「精簡後的有用上下文」一併作為 `extra_context` 輸入：
    # - 若 `QueryExpandingTool` 推薦語意檢索（如：概念性問題、條文定義），請再次使用 `Stage1DecisionTool`。
    # - 若 `QueryExpandingTool` 推薦關鍵詞組，請改用 `KeywordBasedRetrieverTool`，並提供關鍵詞清單進行直接比對查找。

    # 步驟五：根據擴充後的新上下文重新回答問題，不管是`Stage1DecisionTool` 或是`KeywordBasedRetrieverTool` 生成的答案，都需要再次評估品質（回到步驟二），每次回答後都需要進行第二步驟的評估，最多進行三輪修正。

    # ---

    # 請以繁體中文依照上面指示逐步完成這項任務，務必確保每個步驟都被執行。

    # 【使用者問題】：
    # {query}
    # """
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





    