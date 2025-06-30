# dammy.py

def make_dic(document, words_num):
    words = [
        {
            "word": "significant",
            "translation": "重要な",
            "cefr": "B2",
            "tfidf": 0.95,
            "type": "adjective",
        },
        {
            "word": "development",
            "translation": "開発",
            "cefr": "B1",
            "tfidf": 0.88,
            "type": "noun",
        },
        {
            "word": "feature",
            "translation": "機能",
            "cefr": "A2",
            "tfidf": 0.72,
            "type": "noun",
        },
    ]

    # TF-IDF値順にソートして上位 words_num 個だけ返す
    sorted_words = sorted(words, key=lambda x: x["tfidf"], reverse=True)
    return sorted_words[:words_num]
