# prompts.py
from langchain.prompts import PromptTemplate

# 第一階段 prompt
# decision_prompt_template = """
# ###Instruction###
# You are a professional assistant specializing in Taiwanese insurance policies. Based on the following insurance clause content and appendix titles, please determine whether the provided information is sufficient to answer the user's question.

# ###Task###
# - Based on the following insurance clause content and appendix titles, determine whether you can directly answer the user's question.


# - If you can answer the question with the provided context, please provide a full response using the format below:
# - The section below may contain 2 to 5 different insurance clause snippets. Identify and extract only the relevant information needed to answer the user's question. Your answer must be based strictly on the information in the provided context. Avoid introducing content that is not explicitly mentioned. If the answer cannot be found, say:「找不到相關內容」.
# - 請提供詳細的回答，不要給簡答

# 回答：
# [Your answer in Traditional Chinese. Do not repeat the user's question.]

# 條文依據：
# [Cite relevant clause or article if applicable. If none, say:「找不到相關內容」.]

# ###Reference Example - DO NOT COPY, FOR STYLE ONLY###
# Example Question:  
# 國泰人壽真漾心安住院醫療終身保險保障範圍包括哪些項目？

# Example Answer:  
# 回答：  
# 保障項目包含：住院醫療保險金、加護病房或燒燙傷病房保險金、祝壽保險金、身故保險金或喪葬費用保險金，以及所繳保險費的退還。  
# 條文依據：  
# 摘要中明確列出上述保障項目，為本商品之主要給付項目。


# - If you cannot answer the question without more information, please reply in this format.
# - You must extract `needed_appendix_titles` only from the provided Appendix Titles list.
# {{ "answerable": false, "needed_appendix_titles": [""] }}


# ###User Question###
# {question}

# ###Context###
# The following insurance clause content was retrieved based on the user's question:

# {context}

# ###Appendix Titles (Potentially Related)###
# {appendix_titles}
# """
# decision_prompt = PromptTemplate(
#     template=decision_prompt_template,
#     input_variables=["question", "context", "appendix_titles"]
# )

# 第二階段 prompt
rag_prompt_template = """
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
{query}
"""

prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "query"]
)

# 指標評分 prompt
# 指標評分（僅依據已提供的 Information Points 檢核答案覆蓋度）
answer_evaluation_prompt_template = """
###Instruction###
You are an evaluator for a RAG insurance assistant.
You will be given:
1) the user's question, and
2) a pre-computed list of **Information Points** (from an upstream tool),
3) the generated **answer**.

###Task###
- Do NOT create new information points and do NOT modify the provided ones.
- For each information point, give a score between 0 and 1.  
- For **each information point**, if the answer touches on or roughly covers the point, assign a high score (close to 1.0). 
- If the answer clearly misses the point, assign 0.  
- No need to be strict; as long as the point is covered, it should receive credit.  
- Write all point descriptions and any brief notes in **Traditional Chinese**.

###Input###
User Question:
{query}

Information Points (JSON array from upstream tool; each has id/description/must_have):
{information_points}

Generated Answer:
{answer}

###Output Format###
Return **only** a JSON array (no code block, no extra text).  
Each element must include: "id", "point", "分數".
Example:
[
  {{ "id": "1", "point": "需判斷是否符合無理賠回饋保險金之給付條件", "must_have": true,  "分數": 1.0 }},
  {{ "id": "2", "point": "需確認保單年度末仍生存之要件", "must_have": false, "分數": 0.5 }}
]
"""

answer_evaluation_prompt = PromptTemplate(
    template=answer_evaluation_prompt_template,
    input_variables=["query", "information_points", "answer"]
)

# query 修正 prompt
query_expanding_prompt_template = """
你是一位保險問答系統的檢索優化助手，請協助完成以下兩項任務：

任務一：過濾無用資訊  
請根據【使用者問題】與【模型回答】，從【目前檢索到的內容】中篩選出有助於回答問題的部分，將無關資訊刪除，只保留對問題可能有幫助的句子或條文。

任務二：生成進一步檢索用的新問題  
根據你對回答內容的分析，若發現回答仍不完整，請思考下一步應該怎麼檢索補足資料。你可以選擇以下兩種方式之一，請擇一輸出即可：
1. 若需要用語意理解查詢補充條文，請提出具體的新問題  
2. 若更適合用條文中可能出現的具體關鍵字詞查找，請列出一組保險條款中可能出現的專有名詞作為關鍵字。
【注意】：關鍵字請勿使用保險產品名稱（例如：「真康順」、「新鍾心滿福」等），而應聚焦於條文中的醫療、傷病、給付或理賠條件等專有名詞（如：「失智症」、「重大傷病」、「日常生活活動扶助」等）。


請依照以下格式輸出：

【精簡後的有用上下文】：
（請保留有用內容）

【下一步建議 - 新的語意檢索問題】：
方式：semantic
（請寫一個更具體、能幫助補足資訊的查詢問題）

或

【下一步建議 - 關鍵詞組】：
方式：keyword
（請列出具體可用於條文比對的關鍵字）


---
【使用者問題】：
{query}

【目前檢索到的內容】：
{context}

【目前模型回答】：
{answer}
"""
query_expanding_prompt = PromptTemplate(
    template=query_expanding_prompt_template,
    input_variables=["query", "context", "answer"]
)

# ── Information Need（知識點拆解）prompt ─────────────────────────────────────
information_need_prompt_template = """
###Instruction###
You are an expert assistant for Taiwanese insurance Q&A.
Given a user question, decompose it into Information Points with two tiers:
1) **Core points** that are directly required to answer the question (must_have = true).
2) **Optional points** that are helpful but not strictly required (must_have = false).

###Guidelines###
- Output **2–6** points in total.
- **Core points (must_have=true)**: 1–3 項，必須直接來自問題文字本身、能構成「完整回答」所需的要素。
- **Optional points (must_have=false)**: 0–3 項，只能在與問題高度相關且能實際補強理解時加入；請勿臆測等待期、文件、例外等未被提及的要素。
- Each point should be short and clear (1 sentence or phrase in Traditional Chinese).
- Do not include assumptions or extra details not mentioned by the user.

###Output Format (IMPORTANT)###
Return **only** a valid JSON array (no code block, no extra text).
Each element must be an object with keys: "id", "description", "must_have".

###Example###
User Question: 「加護病房保險金有哪些給付範圍？」
Output:
[
  {{
    "id": "1",
    "description": "確認是否有加護病房保險金保障",
    "must_have": true
  }},
  {{
    "id": "2",
    "description": "確認加護病房保險金之給付範圍/條件",
    "must_have": true
  }},
  {{
    "id": "3",
    "description": "（如問題涉及）確認是否包含燒燙傷病房之適用",
    "must_have": false
  }}
]

###User Question###
{query}
"""

information_need_prompt = PromptTemplate(
    template=information_need_prompt_template,
    input_variables=["query"]
)