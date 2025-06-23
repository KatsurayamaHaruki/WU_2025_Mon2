def make_dic(document):
    words = {
    0: {
        "word": "apple",
        "translation": "リンゴ",
        "cefr": "A1", 
        "tfidf": 0.6,
        "type": "noun",
    },
    1: {
        "word": "orange",
        "translation": "オレンジ",
        "cefr": "A2", 
        "tfidf": 0.9,
        "type": "noun",
    }
    }
    return words

print(make_dic("I hava pen")[1]["word"])