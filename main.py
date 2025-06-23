import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from wordfreq import word_frequency
import json
from PyDictionary import PyDictionary # PyDictionaryをインポート

# PyDictionaryのインスタンスを初期化
# これにより、定義や翻訳などの機能が利用可能になります。
dictionary = PyDictionary()

def get_cefr_level(word: str) -> str:
    """
    単語の頻度に基づいてCEFRレベルを推定します。
    これはあくまで簡易的な推定であり、正確なCEFRレベルではありません。
    """
    freq = word_frequency(word, 'en')

    if freq > 0.0001:
        return "A1"
    elif freq > 0.00001:
        return "A2"
    elif freq > 0.000001:
        return "B1"
    elif freq > 0.0000001:
        return "B2"
    elif freq > 0.00000001:
        return "C1"
    else:
        return "C2"

def make_dic(text: str, num_words: int) -> list:
    """
    与えられた英語の文章から重要な単語を抽出し、
    和訳、CEFRレベル、TF-IDF値、品詞名を含むJSON形式のリストを作成します。

    Args:
        text (str): 分析対象の英語の文章。
        num_words (int): 出力する単語の数。デフォルトは10。

    Returns:
        list: 抽出された単語情報を含む辞書のリスト。
    """

    # ① 形態素解析 (spaCy)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # ② ストップワードの除去、記号、数字、空白の除去、名詞・動詞・形容詞・副詞のみを対象
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                filtered_tokens.append(token.lemma_) # 原形を使用

    processed_text = " ".join(filtered_tokens)

    if not processed_text.strip():
        print("ストップワード除去後、有効な単語が残っていません。")
        return []

    # ③ 単語ごとにTF-IDF値を算出
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    word_data_list = []
    for word, score in zip(feature_names, tfidf_scores):
        original_token = next((t for t in doc if t.lemma_ == word and t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]), None)
        if original_token:
            word_data_list.append({
                "word": word,
                "tfidf": score,
                "type": original_token.pos_.lower() # 品詞名を小文字に
            })

    # TF-IDF値が大きい順にソートし、上位N個を取得
    df = pd.DataFrame(word_data_list)
    if df.empty:
        return []

    df = df.sort_values(by="tfidf", ascending=False).head(num_words)
    df = df.reset_index(drop=True)

    final_output = []

    # PyDictionaryで和訳
    for _, row in df.iterrows():
        translated_text = "翻訳エラー (オンライン接続またはライブラリの問題)"
        try:
            # PyDictionaryのtranslateメソッドを使用
            # 日本語は 'ja' または 'Japanese' を指定
            # PyDictionaryの翻訳は内部でオンラインAPIを使用します
            translation = dictionary.translate(row["word"], 'ja')
            if translation:
                translated_text = translation
            else:
                translated_text = "和訳なし" # 翻訳結果がない場合
        except Exception as e:
            print(f"単語 '{row['word']}' の翻訳中にエラーが発生しました: {e}")

        # CEFRレベルの取得
        cefr_level = get_cefr_level(row["word"])

        final_output.append({
            "word": row["word"],
            "translation": translated_text,
            "cefr": cefr_level,
            "tfidf": round(row["tfidf"], 4), # TF-IDF値を小数点以下4桁に丸める
            "type": row["type"],
        })
    
    return final_output

# 例: 使用方法
if __name__ == "__main__":
    example_text = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, and generate human language. It combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. NLP applications include machine translation, spam detection, sentiment analysis, and text summarization. The goal is to bridge the gap between human communication and computer understanding, making interactions more natural and intuitive.
    """

    print("--- 最初の文章での例 ---")
    vocabulary_json = create_vocabulary_json(example_text, num_words=10)
    if vocabulary_json:
        print(json.dumps(vocabulary_json, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")

    print("\n--- 別の文章での例 ---")
    another_text = """
    The quick brown fox jumps over the lazy dog. This is a very common sentence used for testing typefaces and displaying samples of fonts. It contains all the letters of the alphabet.
    """
    another_vocabulary_json = create_vocabulary_json(another_text, num_words=5)
    if another_vocabulary_json:
        print(json.dumps(another_vocabulary_json, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")

    print("\n--- 短い文章での例 ---")
    short_text = "I love programming in Python. It's really fun and useful."
    short_vocabulary_json = create_vocabulary_json(short_text, num_words=3)
    if short_vocabulary_json:
        print(json.dumps(short_vocabulary_json, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")