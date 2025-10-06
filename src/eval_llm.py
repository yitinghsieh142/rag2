import os
import re
import jieba
import sys
from typing import List, Tuple
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 設定 LLM
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

# 設定 prompt 
prompt_template = """
You are an insurance Q&A evaluation expert in Taiwan.

Please evaluate the following information:
- Question: {question}
- Ground Truth (Standard Answer): {ground_truth}
- Retrieved Context: {context}
- Generated Answer (by GPT): {response}

Evaluate from four perspectives. For each aspect, assign a score between 0 and 1 based on the criteria provided, and briefly explain your reasoning:
1. Truthfulness:
    - Score from 0 to 1, where:
    - 0 = completely false or contradictory
    - 1 = completely truthful and accurate
    - Evaluate whether the generated answer factually aligns with the ground truth.
2. Completeness:
    - Score from 0 to 1, where:
    - 0 = missing critical information
    - 1 = covers all important points
    - Evaluate the completeness of the following response compared to the ground truth.
3. Context Faithfulness:
    -Score from 0 to 1, where:
    - 0 = contains statements contradicting or unsupported by the context
    - 1 = All statements are supported by the context
    - Evaluate if the following response is factually consistent with the provided context.

4. Source Relevance:
    - Score from 0 to 1, where:
    - 0 = Completely irrelevant
    - 1 = Highly relevant with key information
    - Evaluate the relevance of each source to answering this question.


Please reply in the following format (in Traditional Chinese):

Truthfulness: <score> （Reason: ）
Completeness: <score> （Reason: ）
Context Faithfulness: <score> （Reason: ）
Source Relevance: <score> （Reason: ）
"""

llmEval_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "ground_truth", "response", "context"]
)

llmEval_chain = LLMChain(llm=client, prompt=llmEval_prompt)

print("Unified LLM-Based 評分模式啟動 (輸入 Enter 離開) \n")

while True:
    sample_question = input("輸入問題 (Question)：").strip()
    if not sample_question:
        print("離開")
        break
    print("請輸入標準答案 (Ground Truth)：（輸入完按 Ctrl+D）：")
    sample_ref_answer = sys.stdin.read().strip()
    print("請輸入檢索到的 Context（輸入完按 Ctrl+D）：")
    sample_context = sys.stdin.read().strip()
    print("請輸入 GPT 生成的回答（輸入完按 Ctrl+D）：")
    sample_generated_answer = sys.stdin.read().strip()

    output = llmEval_chain.run(
        question=sample_question,
        ground_truth=sample_ref_answer,
        context=sample_context,
        response=sample_generated_answer
    )

    # ===== 顯示結果 =====
    print("\n評分結果：")
    print(output)
    # 用正則表達式抓出四個分數
    scores = re.findall(r"(Truthfulness|Completeness|Context Faithfulness|Source Relevance):\s*([01](?:\.\d+)?)", output)

    # 將文字轉為浮點數
    numeric_scores = [float(score[1]) for score in scores]

    if numeric_scores:
        average_score = sum(numeric_scores) / len(numeric_scores)
        print(f"\n🔸 平均分數：{average_score:.2f}")
    else:
        print("\n⚠️ 無法解析分數，請確認格式正確。")

    print("=" * 60)