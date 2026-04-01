import os
import re
import sys
import ast
import pandas as pd
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
    return hallucination_ratio, completeness_ratio, irrelevant_ratio, relevance_scores, irrelevant_scores, wrong_scores

def parse_keypoints_cell(cell_value) -> List[str]:
    """
    支援以下格式：
    1. Python list 字串: "['A', 'B']"
    2. 真正的 list
    3. 多行文字:
       1. A
       2. B
    4. 單一字串
    """
    if pd.isna(cell_value):
        return []

    if isinstance(cell_value, list):
        return [str(x).strip() for x in cell_value if str(x).strip()]

    text = str(cell_value).strip()
    if not text:
        return []

    # 嘗試當成 Python list parse
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # 嘗試多行格式
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        cleaned_lines = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)  # 去掉 1. / 1)
            cleaned_lines.append(line.strip())
        return [x for x in cleaned_lines if x]

    # 單一字串
    return [text]

def score_single_case(question: str, generated_answer: str, keypoints_list: List[str]) -> Tuple[str, float, float, float]:
    keypoints_formatted = list_to_keypoints_str(keypoints_list)
    keypoints_num = len(keypoints_list)

    score_output = score_chain.run(
        question=question,
        generated_answer=generated_answer,
        keypoints=keypoints_formatted,
        keypoints_num=keypoints_num
    )

    hallucination, completeness, irrelevance, relevance_scores, irrelevant_scores, wrong_scores = calculate_evaluation_ratios(
        [score_output], keypoints_num
    )

    output_text = (
        f"Relevance Scores: {relevance_scores}\n"
        f"Irrelevance Scores: {irrelevant_scores}\n"
        f"Wrongness Scores: {wrong_scores}"
    )

    llm_reasoning = score_output

    return score_output, completeness, hallucination, irrelevance, output_text, llm_reasoning


# =========================
# 單題模式
# =========================
def run_single_mode():
    print("保險問答 LLM 單題評估模式，輸入 enter 可離開\n")
    print("請直接貼上問題、回答、keypoints\n")

    while True:
        sample_question = input("輸入問題：").strip()
        if not sample_question:
            print("離開")
            break

        generated_answer = input("輸入回答：").strip()
        if not generated_answer:
            print("❌ 回答不可為空")
            continue

        keypoints_raw = input("輸入 keypoints（Python list 格式，例如 ['A', 'B']）：").strip()
        try:
            keypoints_list = ast.literal_eval(keypoints_raw)
            if not isinstance(keypoints_list, list):
                raise ValueError
        except Exception:
            print("❌ keypoints 格式錯誤，請確保是 Python list 格式，例如：['A', 'B']")
            continue

        score_output, completeness, hallucination, irrelevance, output_text, llm_reasoning = score_single_case(
            sample_question,
            generated_answer,
            keypoints_list
        )

        print("\n===== GPT 評分結果 =====")
        print(score_output)
        print(f"完整率 completeness: {completeness:.4f}")
        print(f"幻覺率 hallucination: {hallucination:.4f}")
        print(f"無關率 irrlevance: {irrelevance:.4f}")
        print("=" * 60)

        print("\n===== output =====")
        print(output_text)

        print("\n===== llm_reasoning =====")
        print(llm_reasoning)


# =========================
# Excel 批次模式
# =========================
def run_excel_mode():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    default_input_path = os.path.join(project_root, "data", "stat_rag4.xlsx")

    print("\nExcel 批次驗證模式")
    user_path = input(f"請輸入 Excel 路徑（直接 Enter 使用預設：{default_input_path}）：").strip()
    input_path = user_path if user_path else default_input_path

    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案：{input_path}")
        return

    print(f"讀取檔案中：{input_path}")
    df = pd.read_excel(input_path)

    required_cols = ["商品代號", "問題", "回答", "Keypoint"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Excel 缺少欄位：{col}")
            return

    # 保留你要的欄名拼法
    if "completeness" not in df.columns:
        df["completeness"] = None
    if "hallucination" not in df.columns:
        df["hallucination"] = None
    if "irrlevance" not in df.columns:
        df["irrlevance"] = None
    if "output" not in df.columns:
        df["output"] = None
    if "llm_reasoning" not in df.columns:
        df["llm_reasoning"] = None

    total = len(df)
    print(f"共 {total} 筆資料，開始評分...\n")

    for idx, row in df.iterrows():
        product_code = row.get("商品代號", "")
        question = "" if pd.isna(row.get("問題")) else str(row.get("問題")).strip()
        answer = "" if pd.isna(row.get("回答")) else str(row.get("回答")).strip()
        keypoints_list = parse_keypoints_cell(row.get("Keypoint"))

        print(f"處理第 {idx + 1}/{total} 筆 | 商品代號: {product_code}")

        if not question or not answer or not keypoints_list:
            print("  ⚠️ 問題 / 回答 / Keypoint 缺值，跳過")
            df.at[idx, "completeness"] = None
            df.at[idx, "hallucination"] = None
            df.at[idx, "irrlevance"] = None
            continue

        try:
            _, completeness, hallucination, irrelevance, output_text, llm_reasoning = score_single_case(
                question,
                answer,
                keypoints_list
            )

            df.at[idx, "completeness"] = round(completeness, 4)
            df.at[idx, "hallucination"] = round(hallucination, 4)
            df.at[idx, "irrlevance"] = round(irrelevance, 4)
            df.at[idx, "output"] = output_text
            df.at[idx, "llm_reasoning"] = llm_reasoning

            print(
                f"  ✅ completeness={completeness:.4f}, "
                f"hallucination={hallucination:.4f}, "
                f"irrlevance={irrelevance:.4f}"
            )

        except Exception as e:
            print(f"  ❌ 第 {idx + 1} 筆失敗：{e}")
            df.at[idx, "completeness"] = None
            df.at[idx, "hallucination"] = None
            df.at[idx, "irrlevance"] = None
            df.at[idx, "output"] = None
            df.at[idx, "llm_reasoning"] = None

    save_choice = input("\n要覆蓋原檔嗎？(y/n): ").strip().lower()

    if save_choice == "y":
        output_path = input_path
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scored{ext}"

    df.to_excel(output_path, index=False)
    print(f"\n✅ 已完成並儲存至：{output_path}")


# =========================
# 主程式入口
# =========================
def main():
    print("請選擇模式：")
    print("1. 單一題目驗證")
    print("2. 整個 Excel 檔案驗證")

    mode = input("輸入 1 或 2：").strip()

    if mode == "1":
        run_single_mode()
    elif mode == "2":
        run_excel_mode()
    else:
        print("❌ 無效輸入，請重新執行")


if __name__ == "__main__":
    main()