import os
import re
import sys
import ast
from typing import List, Tuple
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

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

### 生成 keypoint 的 prompt，暫時不需使用
# keypoint_prompt_template = """
# You are a professional insurance expert in Taiwan.
# In this task, you will be given a question and a standard answer. Based on the standard
# answer, you need to summarize 2 to 5 key points necessary to answer the question.
# These key points should:

# 1. Be directly relevant to the question.
# 2. Be factual and self-contained.
# 3. Clearly reflect the main facts or logic of the standard answer.
# 4. Be written in clear and concise Traditional Chinese sentences.
# 5. Do not include any information not found in the standard answer.

# ---

# Example:
# Question: 國泰人壽真漾心安住院醫療終身保險燒燙傷病房保險金怎麼申領?

# Standard Answer: 
# 需求文件：
# 1. 保險單或其謄本
# 2. 保險金申請書
# 3. 醫療診斷書或住院證明。申請「加護病房或燒燙傷病房保險金」者，須列明進、出加護病房或
# 燒燙傷病房日期。（但要保人或被保險人為醫師時，不得為被保險人出具診斷書或住院證
# 明。）
# 4. 受益人的身分證明

# Key Points:
# 1. 申請燒燙傷病房保險金時需附上保險單或其謄本。
# 2. 須提供完整的保險金申請書。
# 3. 須檢附列明進出燒燙傷病房日期的醫療診斷書或住院證明。
# 4. 若要保人或被保險人為醫師，不得為被保險人自行出具診斷書或住院證明。
# 5. 受益人需提供身分證明文件。

# ---

# Question: {question}

# Standard Answer: {ground_truth}

# Key Points:
# 1.
# """

# keypoint_prompt = PromptTemplate(
#     template=keypoint_prompt_template,
#     input_variables=["question", "ground_truth"]
# )

# keypoint_chain = LLMChain(llm=client, prompt=keypoint_prompt)



### 評分 prompt 輸入：問題、回答、keypoint 做使用
score_prompt_template = """
You are a professional insurance evaluator in Taiwan.

In this task, you will receive a question, a generated answer, and multiple key points from a standard answer.
You must evaluate whether each key point is properly addressed in the generated answer.

For each key point, provide:
- A short analysis in Traditional Chinese.
- Three scores (each from 0 to 1) for:
  - Relevance Score: how well the generated answer matches the key point
  - Irrelevance Score: how unrelated the answer is to the key point
  - Wrongness Score: how factually or logically incorrect the answer is
- Scores must add up to 1.0 (you can assign 1.0 to one, or split values like 0.7, 0.2, 0.1).

---

Scoring Guidelines:
- Relevance Score: High if the answer accurately covers the key point, even if phrased differently.
- Irrelevance Score: High if the answer does not mention the topic at all.
- Wrongness Score: High ONLY if the answer contradicts the key point or states a wrong fact.
- If the answer is accurate but phrased differently, set Wrongness Score to 0.0. Do NOT penalize wording differences.
- If the answer partially covers the key point (missing details) but has no contradiction, split between Relevance and Irrelevance; keep Wrongness at 0.0.
- If the answer fully and correctly addresses the key point, use (Relevance, Irrelevance, Wrongness) = (1.0, 0.0, 0.0).


DO NOT classify into a single label. Only assign scores.
DO NOT assign a single label. Only assign scores.
[[[Wrongness Score]]] should only receive a high value when there is a specific factual or logical conflict between the key point and the generated answer. If key content is missing, increase the [[[Irrelevance Score]]] instead.
[[[Relevance Score]]] does not require the generated answer to include all details. It is enough to reflect the main idea of the key point, even with different wording or partial coverage.
Make sure the number of key points evaluated exactly matches the number listed. Do not skip, merge, or duplicate any key points.
If the generated answer addresses the correct topic but gives incorrect information, assign a higher [[[Wrongness Score]]]. If it doesn’t mention the topic at all, assign a higher [[[Irrelevance Score]]].
---

Test Case:
Question: {question}

Generated Answer:
{generated_answer}

Standard Answer Key Points:
Here are {keypoints_num} key points:
{keypoints}

Key Point Evaluation:
For each key point, use the following format:

Key Point {keypoints_num}: {keypoints}
分析：
Relevance Score: 0.0 to 1.0
Irrelevance Score: 0.0 to 1.0
Wrongness Score: 0.0 to 1.0
"""

score_prompt = PromptTemplate(
    template=score_prompt_template,
    input_variables=["question", "generated_answer", "keypoints", "keypoints_num"]
)

score_chain = LLMChain(llm=client, prompt=score_prompt)


# ===== 工具函式 =====
def normalize_numbers(s: str) -> str:
    trans_table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return s.translate(trans_table)

# def extract_scores(text: str, label: str) -> list[float]:
#     pattern = rf"{label}\s*[:：]?\s*\*{{0,2}}\s*([0-9０１２３４５６７８９．.]+)"
#     matches = re.findall(pattern, text)
#     return [float(normalize_numbers(m)) for m in matches]
def extract_scores(text: str, label: str) -> list[float]:
    # 1) 清掉粗體與常見列表符號
    cleaned = text.replace("**", "")
    cleaned = re.sub(r"^[\s>*-]+\s*", "", cleaned, flags=re.MULTILINE)

    # 2) 容忍半形/全形冒號
    pattern = rf"(?mi)^\s*[-*•>–—]?\s*{re.escape(label)}\s*[:：]\s*\**\s*([0-9０-９.．]+)\s*$"
    matches = re.findall(pattern, cleaned)
    return [float(normalize_numbers(m)) for m in matches]

def list_to_keypoints_str(kp_list: List[str]) -> str:
    return "\n".join([f"{i+1}. {kp.strip()}" for i, kp in enumerate(kp_list)])

def calculate_evaluation_ratios(model_responses: List[str], keypoints_num: int) -> Tuple[float, float, float]:
    relevance_scores = []
    irrelevant_scores = []
    wrong_scores = []

    for response in model_responses:
        relevance_scores += extract_scores(response, "Relevance Score")
        irrelevant_scores += extract_scores(response, "Irrelevance Score")
        wrong_scores += extract_scores(response, "Wrongness Score")

    print("Matches (Relevance):", re.findall(r"Relevance Score.*", response))
    print("Matches (Wrongness):", re.findall(r"Wrongness Score.*", response))
    print("Matches (Irrelevance):", re.findall(r"Irrelevance Score.*", response))
    print("Relevance Scores:", relevance_scores)
    print("Irrelevance Scores:", irrelevant_scores)
    print("Wrongness Scores:", wrong_scores)

    def pad_scores(scores):
        if len(scores) >= keypoints_num:
            return scores[:keypoints_num]
        avg = sum(scores) / len(scores) if scores else 0.0
        return scores + [avg] * (keypoints_num - len(scores))

    relevance_scores = pad_scores(relevance_scores)
    irrelevant_scores = pad_scores(irrelevant_scores)
    wrong_scores = pad_scores(wrong_scores)

    completeness_ratio = sum(relevance_scores) / keypoints_num
    hallucination_ratio = sum(wrong_scores) / keypoints_num
    irrelevant_ratio = sum(irrelevant_scores) / keypoints_num

    print(f"➡️ keypoints_num: {keypoints_num}")
    return hallucination_ratio, completeness_ratio, irrelevant_ratio


# === 啟動互動模式 ===
print("保險問答 LLM 評估模式，輸入enter可離開\\n")
print("輸入順序：問題 → key points（多行）→ 生成答案（多行）\n")
while True:
    sample_question = input("輸入問題：").strip()
    if not sample_question:
        print("離開")
        break

    print("🔹 請貼上 keypoints（list 格式）：")
    keypoints_str = sys.stdin.read().strip()
    try:
        keypoints_list = ast.literal_eval(keypoints_str)
        if not isinstance(keypoints_list, list):
            raise ValueError
    except Exception:
        print("❌ keypoints 格式錯誤，請確保是 Python list 格式，例如：['A', 'B']")
        continue

    keypoints_formatted = list_to_keypoints_str(keypoints_list)
    keypoints_num = len(keypoints_list)

    print("🔹 輸入 GPT 生成答案（輸入完請按 Ctrl+D）：")
    generated_answer = sys.stdin.read().strip()

    # 呼叫 GPT 評分
    score_output = score_chain.run(
        question=sample_question,
        generated_answer=generated_answer,
        keypoints=keypoints_formatted,
        keypoints_num=keypoints_num
    )

    print("\n===== GPT 評分結果 =====")
    print(score_output)

    hallucination, completeness, irrelevance = calculate_evaluation_ratios([score_output], keypoints_num)
    print(f"幻覺率: {hallucination:.2f} | 完整率: {completeness:.2f} | 無關率: {irrelevance:.2f}")
    print("="*60)
