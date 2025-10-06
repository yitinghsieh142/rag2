import os
import json
import re
import pandas as pd


# === 基本設定 ===
output_folder = "../output"
# 跑全部
# product_folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
# 跑特定檔案
product_folders = ["CFG"]

output_links_folder = "../soft_links_output"
os.makedirs(output_links_folder, exist_ok=True)



# === 關鍵詞擷取函數 ===
def normalize_text(s):
    s = s.replace("　", "")  # 全形空格
    s = re.sub(r"[「」『』【】（）()、《》．・—：:；;，,。！？!?.\"'\[\]\{\}]", "", s)  # 標點
    return s.strip()

def extract_keywords(title):
    # 清理標題：移除「附表一：」等前綴 + 標點
    clean_title = re.sub(r"^附[表件](?:[一二三四五六七八九十0-9]*)?[：:]*", "", title)
    clean_title = normalize_text(clean_title)
    print(clean_title)

    # 定義停用詞（無意義連接詞）
    stopwords = {"與", "及", "和", "之", "的", "與其", "或"}

    # 將標題詞彙化，這裡每個詞是連續2~5字的中文字（包含中間有「與」的）
    raw_words = re.findall(r"[\u4e00-\u9fff]{2,5}", clean_title)

    # 強制切分詞 → 若某個詞包含 stopword，就把它拆開
    split_words = []
    for word in raw_words:
        parts = re.split(r"[與及和之的與其或]", word)
        split_words.extend([w for w in parts if w])  # 移除空白

    # 過濾 + 去掉尾巴是「表」的詞
    words = []
    for w in split_words:
        if w and w not in stopwords:
            if w.endswith("表") and len(w) > 2:
                words.append(w[:-1])  # 去掉「表」
            else:
                words.append(w)
    
    # 建立 1-gram 與 2-gram 組合
    phrases = set()
    for i in range(len(words)):
        phrases.add(words[i])  # 1-gram
        if i + 1 < len(words):
            phrases.add(words[i] + words[i + 1])  # 2-gram
    return list(phrases)


# === 主邏輯：處理單一 JSON 檔 ===
def process_single_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("sections", [])
    appendices = data.get("appendices", [])
    soft_links = []

    for appendix in appendices:
        appendix_id = appendix['chunk_id']
        appendix_title = appendix['title']
        keywords = extract_keywords(appendix_title)
        print(appendix_title)
        print(keywords)

        if not keywords:
            continue

        for sec in sections:
            sec_id = sec["chunk_id"]
            sec_title = sec["title"]
            content = sec_title + sec["content"]
            normalized_content = normalize_text(content)
            matched = [kw for kw in keywords if normalize_text(kw) in normalized_content]

            if matched:
                soft_links.append({
                    "appendix_chunk_id": appendix_id,
                    "appendix_title": appendix_title,
                    "section_chunk_id": sec_id,
                    "section_title": sec_title,
                    "matched_keywords": matched
                })

    return soft_links

# === 處理所有 JSON ===
for folder in product_folders:
    folder_path = os.path.join(output_folder, folder)
    json_path = os.path.join(folder_path, "pdf_parsing.json")

    if os.path.exists(json_path):
        soft_links = process_single_json(json_path)

        if soft_links:
            json_out_path = os.path.join(output_links_folder, f"{folder}.json")
            compressed_links = {}
            for link in soft_links:
                a_id = str(link["appendix_chunk_id"])
                s_id = link["section_chunk_id"]
                if a_id not in compressed_links:
                    compressed_links[a_id] = []
                compressed_links[a_id].append(s_id)

            # 移除重複 section id 並排序（可選）
            for a_id in compressed_links:
                compressed_links[a_id] = sorted(list(set(compressed_links[a_id])))

            with open(json_out_path, "w", encoding="utf-8") as f_out:
                json.dump(compressed_links, f_out, ensure_ascii=False, indent=2)
            print(f"{folder}/pdf_parsing.json → 共 {len(soft_links)} 筆 soft link → 輸出到 {json_out_path}")
        else:
            print(f"⚠ 無語意對應：{folder}/pdf_parsing.json")
    else:
        print(f"找不到：{folder}/pdf_parsing.json")