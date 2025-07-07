import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
from googletrans import Translator

# --- グローバル設定: ライブラリとデータファイルの初期化 ---
print("--- ライブラリとデータモデルを初期化中 ---")

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
    df_a1_b2 = pd.read_csv("cefrj-vocabulary-profile-1.51.csv", encoding="shift-jis")
    df_c1_c2 = pd.read_csv("octanove-vocabulary-profile-c1c2-1.0.csv", encoding="shift-jis")
    COMBINED_CEFR_DF = pd.concat([df_a1_b2, df_c1_c2], ignore_index=True)
    COMBINED_CEFR_DF['headword'] = COMBINED_CEFR_DF['headword'].str.lower()
    COMBINED_CEFR_DF.set_index('headword', inplace=True)
    print("CEFRのCSVファイルを正常に読み込み、結合しました。")
except FileNotFoundError as e:
    print(f"エラー: CEFRのCSVファイルが見つかりません。パスを確認してください。: {e}")
    COMBINED_CEFR_DF = None
except Exception as e:
    print(f"CSVファイルの読み込み中に予期せぬエラーが発生しました。: {e}")
    COMBINED_CEFR_DF = None

print("--- 初期化完了 ---")


def get_cefr_level(word: str) -> str:
    """
    事前読み込みしたデータフレームから単語のCEFRレベルを高速に検索します。
    """
    if COMBINED_CEFR_DF is None:
        return "CEFRデータ未読込"
    try:
        # 'cefr'列を指定するように修正（元のコードでは'CEFR'になっていたため）
        level = COMBINED_CEFR_DF.loc[word.lower(), 'CEFR']
        if isinstance(level, pd.Series):
            return level.iloc[0]
        return level
    except KeyError:
        return "N/A"

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
        return pd.DataFrame() # 空のデータフレームを返すように統一
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


# ★★★★★ 機能改善: CEFRレベルでフィルタリングする機能を追加 ★★★★★
def make_dic(tfidf_df: pd.DataFrame, num_words: int) -> list:
    """
    TF-IDFデータフレームからCEFRレベルがA1, A2の単語を除外し、
    上位N個の単語で和訳などを付与した辞書リストを作成します。
    """
    if tfidf_df.empty:
        return []

    # 1. 全単語のCEFRレベルを取得し、新しい列としてデータフレームに追加
    tfidf_df['cefr'] = tfidf_df['word'].apply(get_cefr_level)

    # 2. CEFRレベルが 'B2','C1','C2'の単語のみ表示
    # isin()でA1,A2の行にTrue/Falseのフラグを立て、'~'でそれを反転させて除外する
    filtered_df = tfidf_df[~tfidf_df['cefr'].isin(['A1', 'A2', 'B1','N/A'])]
    
    # 3. フィルタリング後のデータフレームをソートし、上位N個を取得
    df_sorted = filtered_df.sort_values(by="tfidf", ascending=False).head(num_words)
    df_sorted = df_sorted.reset_index(drop=True)

    final_output = []
    for _, row in df_sorted.iterrows():
        translated_text = "翻訳エラー"
        try:
            translation = translator.translate(row["word"], dest='ja')
            translated_text = translation.text if translation else "和訳なし"
        except Exception as e:
            print(f"単語 '{row['word']}' の翻訳中にエラーが発生しました: {e}")
        
        # 4. CEFRレベルはすでにあるので、get_cefr_levelを再度呼ばず、そのまま使う
        cefr_level = row["cefr"]
        
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

# ★★★★★ 構成改善: メインブロックの処理の流れを整理 ★★★★★
if __name__ == "__main__":
    example_text = """
In the dead of night, Melos's childhood friend, Selinuntiou, was summoned to the royal castle. Before the tyrant Dionysius, the good friends reunited after two years. Melos shared all his circumstances with his friend. Selinuntiou nodded silently and embraced Melos tightly. That was enough for the bond between friends. Selinuntiou was bound with ropes. Melos set off immediately. It was early summer, with a sky full of stars.
That night, Melos rushed the ten miles without sleeping at all, arriving in the village the next morning, with the sun already high in the sky. The villagers were already out in the fields beginning their work. Melos's sixteen-year-old sister was watching over the sheep in place of her brother that day. She was startled to see her brother stumbling towards her, looking utterly exhausted. She bombarded him with questions.
"It's nothing," Melos tried to force a smile. "I left some business in the city. I have to go back there soon. Tomorrow, we will hold your wedding. It’s better to do it sooner."
His sister blushed.
"Are you happy? I bought a beautiful dress too. Now, go and tell the villagers that the wedding is tomorrow."
Melos staggered again and returned home to decorate the altar for the gods, set up the banquet table, and soon collapsed on the floor, falling into a deep sleep so profound that he could hardly breathe.
He woke up at night. Immediately after waking, Melos visited the groom's house. He then requested to postpone the wedding to the next day, explaining that there were some circumstances. The groom, a shepherd, was surprised and replied that it couldn't be done as they hadn't prepared anything yet and asked him to wait until the grape harvest season. Melos insisted that he could not wait and begged him to change it to tomorrow. The groom was stubborn and did not easily agree. They argued until dawn, and finally, Melos managed to placate him, persuade him, and convince him. The wedding took place at noon. Just as the bride and groom were making their vows to the gods, dark clouds covered the sky, and it started to rain lightly, eventually turning into a torrential downpour. The villagers attending the feast felt a sense of foreboding, but they each tried to keep their spirits up, enduring the stifling heat of the small house, singing cheerfully and clapping their hands. Melos, too, was beaming with joy and for a moment forgot about his promise to the king. As the night wore on, the celebration became more lively and extravagant, and the people completely disregarded the heavy rain outside. Melos thought to himself that he wanted to stay there forever, wishing to live his life among these wonderful people, but at that moment, he was not in control of his own fate. It was a frustrating situation. Melos lashed himself for his own desires and ultimately made the decision to depart. There was still plenty of time before sunset the next day. He thought about taking a short nap and then setting off right away. By that time, the rain should have eased up. He wanted to linger in that house for just a little longer. Even a man like Melos had lingering feelings. He approached the bride, who seemed dazed and intoxicated with joy tonight.
"Congratulations. I'm feeling tired, so I'll take a little break and sleep. When I wake up, I'll head to the city right away. I have something important to attend to. Even if I'm not there, you have a kind husband now, so you won't be lonely at all. Your brother's greatest dislikes are being suspicious of others and telling lies. You know that, right? You must not keep any secrets from your husband. That's all I wanted to say. Your brother is probably a great man, so you should take pride in that."
The bride nodded dreamily. Melos then patted the groom on the shoulder,
"We both lack preparation. In my house, the only treasures are my sister and a sheep. I will give you everything I have. Also, please take pride in having become Melos' brother."""

# --- 最初の文章で実行 ---
    print("\n--- 最初の文章で実行 ---")
    print("TF-IDF値を計算中...")
    # 1. TF-IDF計算を先に行い、結果をデータフレームとして保持
    tfidf_dataframe_long = preprocess_and_calculate_tfidf(example_text)
    
    print("計算結果からA1/A2を除外した上位30単語の辞書を生成中...")
    # 2. データフレームを関数に渡す
    vocabulary_json_30 = make_dic(tfidf_dataframe_long, num_words=30)
    if vocabulary_json_30:
        print(json.dumps(vocabulary_json_30, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")

    print("\n同じ文章からA1/A2を除外した上位5単語の辞書を生成中...")
    # 3. 同じデータフレームを再利用して、単語数だけ変えて再度実行
    vocabulary_json_5 = make_dic(tfidf_dataframe_long, num_words=5)
    if vocabulary_json_5:
        print(json.dumps(vocabulary_json_5, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")
    
    
    # --- 別の短い文章で実行 ---
    print("\n\n--- 別の短い文章で実行 ---")
    short_text = "I love programming in Python. It's really fun and useful."
    print("TF-IDF値を計算中...")
    tfidf_dataframe_short = preprocess_and_calculate_tfidf(short_text)
    
    print("計算結果からA1/A2を除外した上位3単語の辞書を生成中...")
    short_vocabulary_json = make_dic(tfidf_dataframe_short, num_words=3)
    if short_vocabulary_json:
        print(json.dumps(short_vocabulary_json, indent=4, ensure_ascii=False))
    else:
        print("単語表が生成されませんでした。")