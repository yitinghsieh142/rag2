from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import AzureChatOpenAI
from langchain.tools import StructuredTool
from tools import stage1_decision_tool, stage2_rag_with_appendix_tool_wrapper, evaluate_answer_metrics, query_expanding_metrics, keyword_retriever_tool
from functools import partial
from utils import extract_prod_id_from_query
from pydantic import BaseModel, Field
from prompt import *
from typing import List, Optional
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

current_product_name = None
def stage1_tool_wrapper(query: str, extra_context: Optional[str] = None):
    return stage1_decision_tool(
        query=query,
        default_product_name=current_product_name,
        extra_context=extra_context
    )

# ======== 設定 LLM ========
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0
)

# ======== 註冊 Tool ========
class EvaluationInput(BaseModel):
    query: str = Field(description="使用者提出的保險問題")
    context: str = Field(description="用於回答的檢索內容（上下文）")
    answer: str = Field(description="模型產生的回覆")

class QueryExpansionInput(BaseModel):
    query: str = Field(description="使用者的原始問題")
    context: str = Field(description="檢索到的上下文內容")
    answer: str = Field(description="根據上下文產生的模型回覆")

class KeywordRetrieverInput(BaseModel):
    query: str = Field(description="使用者提出的保險問題")
    keywords: List[str] = Field(description="關鍵字清單，例如 ['失智症', '身故']")
    extra_context: Optional[str] = Field(default=None, description="從上次回答中留下的精簡上下文")


class Stage1DecisionInput(BaseModel):
    query: str = Field(description="使用者提出的保險問題")
    extra_context: Optional[str] = Field(default=None, description="從上次回答中留下的精簡上下文")

# decision_tool = Tool(
#     name="Stage1DecisionTool",
#     func=lambda query: stage1_decision_tool(query, default_product_name=current_product_name),
#     description="使用初始 context 回答保險問題"
# )

decision_tool = StructuredTool.from_function(
    name="Stage1DecisionTool",
    func=stage1_tool_wrapper,
    description="使用初始 context（包含選擇性額外上下文 extra_context）回答保險問題，若有上輪留下的【精簡後有用上下文】，請務必放入 extra_context 欄位中",
    args_schema=Stage1DecisionInput
)

# decision_tool = Tool(
#     name="Stage1DecisionTool",
#     func=stage1_tool_wrapper,
#     description="使用初始 context（包含選擇性額外上下文 extra_context）回答保險問題，若有上輪留下的【精簡後有用上下文】，請一定要放入 extra_context 欄位中",
# )

answer_evaluator_tool = StructuredTool.from_function(
    name="AnswerEvaluatorTool",
    func=evaluate_answer_metrics,
    description=(
        "針對保險問題的回答進行評估："
        "請列出回答此問題所需的關鍵資訊點（用繁體中文描述），"
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

keyword_tool = StructuredTool.from_function(
    name="KeywordBasedRetrieverTool",
    func=keyword_retriever_tool,
    description="使用初始 context（包含選擇性額外上下文 extra_context）回答保險問題，若有上輪留下的【精簡後有用上下文】，請一定要放入 extra_context 欄位中",
    args_schema=KeywordRetrieverInput
)

tools = [decision_tool, answer_evaluator_tool, query_expanding_tool, keyword_tool]

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
    # wrapped_input = (
    #     "請依序完成以下任務：\n"
    #     "1. 使用合適的工具回答使用者的保險問題。\n"
    #     "2. 每次生成回答後，請使用評估工具從四個面向評估你的答案品質（Context Relevancy、Context Adherence、Answer Relevancy、Grading Note）。\n"
    #     "請確保格式為：答案 + 條文依據。\n\n"
    #     f"使用者問題：{query}"
    # )
    wrapped_input = f"""
    你是一位保險問答助手，請依照以下流程一步步完成，不能跳過：

    步驟一：使用檢索內容回答使用者的保險問題，產出「答案 + 條文依據」。

    步驟二：使用工具 `AnswerEvaluatorTool` 列出使用者問題中應包含的關鍵資訊點，並針對每個資訊點給出 0～1 的分數，用來評估目前回答是否完整。

    步驟三：如果任何一項資訊點評分低於 0.8，請使用 `QueryExpandingTool` 根據「原始問題 + 上下文 + 回答」找出缺乏的資訊，並產生新的語意問題或是關鍵詞組們進行進一步檢索。

    步驟四：請根據 `QueryExpandingTool` 產生的新問題或是關鍵詞組，執行進一步的檢索，並將上一步的「精簡後的有用上下文」一併作為 `extra_context` 輸入：
    - 若 `QueryExpandingTool` 推薦語意檢索（如：概念性問題、條文定義），請再次使用 `Stage1DecisionTool`。
    - 若 `QueryExpandingTool` 推薦關鍵詞組，請改用 `KeywordBasedRetrieverTool`，並提供關鍵詞清單進行直接比對查找。

    步驟五：根據擴充後的新上下文重新回答問題，不管是`Stage1DecisionTool` 或是`KeywordBasedRetrieverTool` 生成的答案，都需要再次評估品質（回到步驟二），每次回答後都需要進行第二步驟的評估，最多進行三輪修正。

    ---

    請以繁體中文依照上面指示逐步完成這項任務，務必確保每個步驟都被執行。

    【使用者問題】：
    {query}
    """
    result = agent.invoke({"input": wrapped_input})





    