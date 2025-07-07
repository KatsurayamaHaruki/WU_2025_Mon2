import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
# ★★★ 変更: PyDictionary の代わりに googletrans をインポート ★★★
from googletrans import Translator

# --- グローバル設定: ライブラリとデータファイルの初期化 ---
print("--- ライブラリとデータモデルを初期化中 ---")

# ★★★ 変更: googletrans の Translator を初期化 ★★★
translator = Translator()

nlp = spacy.load("en_core_web_sm")

POS_MAP_EN_TO_JA = {
    "NOUN": "名詞",
    "VERB": "動詞",
    "ADJ": "形容詞",
    "ADV": "副詞",
    "PROPN": "固有名詞"
}

try:
    # (CSVファイルの読み込み部分は変更なし)
    df_a1_b2 = pd.read_csv("cefrj-vocabulary-profile-1.51.csv", encoding="shift-jis")
    df_c1_c2 = pd.read_csv("octanove-vocabulary-profile-c1c2-1.0.csv", encoding="shift-jis")
    COMBINED_CEFR_DF = pd.concat([df_a1_b2, df_c1_c2], ignore_index=True)
    # ★★★ 改善: 検索を安定させるため、インデックスをすべて小文字に変換 ★★★
    COMBINED_CEFR_DF['headword'] = COMBINED_CEFR_DF['headword'].str.lower()
    COMBINED_CEFR_DF.set_index('headword', inplace=True)
    print(COMBINED_CEFR_DF)
    print("CEFRのCSVファイルを正常に読み込み、結合しました。")
except FileNotFoundError as e:
    print(f"エラー: CEFRのCSVファイルが見つかりません。パスを確認してください。: {e}")
    COMBINED_CEFR_DF = None
except Exception as e:
    print(f"CSVファイルの読み込み中に予期せぬエラーが発生しました: {e}")
    COMBINED_CEFR_DF = None

print("--- 初期化完了 ---")


def get_cefr_level(word: str) -> str:
    """
    事前読み込みしたデータフレームから単語のCEFRレベルを高速に検索します。
    """
    if COMBINED_CEFR_DF is None:
        return "CEFRデータ未読込"
    try:
        # ★★★ 改善: 検索する単語を小文字に変換して、大文字/小文字の差異を吸収 ★★★
        level = COMBINED_CEFR_DF.loc[word.lower(), 'CEFR']
        if isinstance(level, pd.Series):
            return level.iloc[0]
        return level
    except KeyError:
        return "N/A"

# (preprocess_and_calculate_tfidf 関数は変更なし)
def preprocess_and_calculate_tfidf(text: str) -> pd.DataFrame:
    doc = nlp(text)
    filtered_tokens = []
    allowed_pos = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num:
            if token.pos_ in allowed_pos:
                filtered_tokens.append(token.lemma_)
    if not filtered_tokens:
        print("有効な単語が見つかりませんでした。")
        return pd.DataFrame(columns=["word", "tfidf", "type"])
    processed_text = " ".join(filtered_tokens)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_to_pos = {token.lemma_: token.pos_ for token in doc if token.lemma_ in feature_names}
    word_data_list = []
    for word, score in zip(feature_names, tfidf_scores):
        word_data_list.append({
            "word": word,
            "tfidf": score,
            "type": word_to_pos.get(word, "unknown")
        })
    return pd.DataFrame(word_data_list)


def make_dic(example_text: str, num_words: int) -> list:
    """
    事前計算されたTF-IDFデータフレームから上位N個の単語を選択し、
    和訳とCEFRレベルを付与してJSON形式のリストを作成します。
    """

    tfidf_df = preprocess_and_calculate_tfidf(example_text)
    if tfidf_df.empty:
        return []

    df_sorted = tfidf_df.sort_values(by="tfidf", ascending=False).head(num_words)
    df_sorted = df_sorted.reset_index(drop=True)

    final_output = []
    for _, row in df_sorted.iterrows():
        translated_text = "翻訳エラー"
        try:
            # ★★★ 変更: googletrans を使って翻訳 ★★★
            translation = translator.translate(row["word"], dest='ja')
            translated_text = translation.text if translation else "和訳なし"
        except Exception as e:
            print(f"単語 '{row['word']}' の翻訳中にエラーが発生しました: {e}")

        cefr_level = get_cefr_level(row["word"])
        
        pos_en = row["type"]
        pos_ja = POS_MAP_EN_TO_JA.get(pos_en, pos_en)

        final_output.append({
            "word": row["word"],
            "translation": translated_text,
            "cefr": cefr_level,
            "tfidf": round(row["tfidf"], 4),
            "type": pos_ja,
        })
    
    return final_output

# (メインの実行ブロックは変更なし)
if __name__ == "__main__":
    example_text = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, and generate human language. It combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. NLP applications include machine translation, spam detection, sentiment analysis, and text summarization. The goal is to bridge the gap between human communication and computer understanding, making interactions more natural and intuitive.
    """
    print("\n--- 最初の文章で実行 ---")
    print("TF-IDF値を計算中...")
    print("計算結果から上位10単語の辞書を生成中...")
    vocabulary_json_10 = make_dic(example_text, num_words=10)
    if vocabulary_json_10:
        print(json.dumps(vocabulary_json_10, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")
    print("\n同じ文章から上位5単語の辞書を生成中（TF-IDFの再計算なし）...")
    vocabulary_json_5 = make_dic(tfidf_dataframe, num_words=5)
    if vocabulary_json_5:
        print(json.dumps(vocabulary_json_5, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")
    print("\n\n--- 別の短い文章で実行 ---")
    short_text = "I love programming in Python. It's really fun and useful."
    print("TF-IDF値を計算中...")
    short_tfidf_dataframe = preprocess_and_calculate_tfidf(short_text)
    print("計算結果から上位3単語の辞書を生成中...")
    short_vocabulary_json = make_dic(short_tfidf_dataframe, num_words=3)
    if short_vocabulary_json:
        print(json.dumps(short_vocabulary_json, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")