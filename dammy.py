def make_dic(document):
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

    return words

print(make_dic("I hava pen")[1]["word"])