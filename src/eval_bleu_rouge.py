import jieba
import pandas as pd
from rouge import Rouge
import rouge_chinese
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ROUGELScore:
    name: str = "ROUGELScore"

    def __init__(self, language: str = "zh"):
        self.rouge = Rouge(metrics=["rouge-l"])
        self.language = language

    def _calculate_rouge_l_score_chinese(self, hypothesis: str, reference: str) -> float:
        rouge = rouge_chinese.Rouge()
        try:
            scores = rouge.get_scores(hypothesis, reference)
            return scores[0]["rouge-l"]["f"]
        except ValueError:
            return 0.0

    def score_one(self, reference: str, prediction: str) -> float:
        if not reference or not prediction:
            return 0.0
        if self.language == "zh":
            reference_cut = ' '.join(jieba.cut(reference))
            prediction_cut = ' '.join(jieba.cut(prediction))
            return self._calculate_rouge_l_score_chinese(prediction_cut, reference_cut)
        return 0.0

    def score_csv(self, csv_path: str, ref_col: str = "測試集回答", pred_col: str = "回答結果") -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df[df[ref_col].notna() & df[pred_col].notna()]
        df = df[(df[ref_col].str.strip() != "") & (df[pred_col].str.strip() != "")]
        scores = []
        for _, row in df.iterrows():
            ref = str(row[ref_col])
            pred = str(row[pred_col])
            score = self.score_one(ref, pred)
            scores.append(score)
        df["rouge-l"] = scores
        print(f"平均 ROUGE-L 分數: {sum(scores)/len(scores):.4f}")
        return df

class BLEUScore:
    name: str = "BLEUScore"

    def __init__(self, language: str = "zh"):
        self.language = language
        self.smooth_fn = SmoothingFunction().method1

    def score_one(self, reference: str, prediction: str) -> float:
        if not reference or not prediction:
            return 0.0
        if self.language == "zh":
            reference_tokens = list(jieba.cut(reference))
            prediction_tokens = list(jieba.cut(prediction))
        else:
            reference_tokens = reference.split()
            prediction_tokens = prediction.split()
        return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=self.smooth_fn)

    def score_csv(self, csv_path: str, ref_col: str = "測試集回答", pred_col: str = "回答結果") -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df[df[ref_col].notna() & df[pred_col].notna()]
        df = df[(df[ref_col].str.strip() != "") & (df[pred_col].str.strip() != "")]
        scores = []
        for _, row in df.iterrows():
            ref = str(row[ref_col])
            pred = str(row[pred_col])
            score = self.score_one(ref, pred)
            scores.append(score)
        df["bleu"] = scores
        print(f"平均 BLEU 分數: {sum(scores)/len(scores):.4f}")
        return df

if __name__ == "__main__":
    # 下一行放入 data 路徑
    csv_path = "../data/scoretest_L66.csv"

    # ROUGE
    rouge_scorer = ROUGELScore()
    result_df = rouge_scorer.score_csv(csv_path)
    result_df.to_csv("../data/test_r_with_rouge.csv", index=False)

    # BLEU
    bleu_scorer = BLEUScore()
    df_bleu = bleu_scorer.score_csv(csv_path)
    df_bleu.to_csv("../data/test_r_with_bleu.csv", index=False)