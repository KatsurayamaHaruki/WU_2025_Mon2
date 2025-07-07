# app.py

from flask import Flask, request, render_template, session
# main.py の関数名を make_dic から create_vocabulary_json に変更したため、ここも修正します。
from main import make_dic as create_vocabulary_json # make_dicはcreate_vocabulary_jsonに名前が変更されたため

import os

app = Flask(__name__)
# 実際のアプリケーションでは、より安全な秘密鍵を使用してください。
# 例: app.secret_key = os.urandom(24)
app.secret_key = 'your_super_secret_key_here' 

@app.route('/')
def index():
    """
    アプリケーションのトップページを表示します。
    """
    return render_template('index.html')

@app.route('/loading', methods=['POST'])
def loading():
    """
    テキスト入力フォームからのデータを受け取り、セッションに保存します。
    その後、処理中の画面（loading.html）を表示します。
    """
    # フォームから送信されたテキストと単語数をセッションに保存
    session['text'] = request.form['text']
    session['top_n'] = int(request.form['top_n'])
    return render_template('loading.html')

@app.route('/result')
def result():
    """
    セッションからテキストと単語数を取得し、TF-IDF分析を実行します。
    分析結果をresult.htmlに渡して表示します。
    """
    # セッションからテキストと単語数を取得。デフォルト値を設定してエラーを回避
    text = session.get('text', '')
    top_n = session.get('top_n', 3) # index.html の初期値に合わせて3に変更しました

    # main.py で定義された create_vocabulary_json 関数を呼び出して単語分析を実行
    words = create_vocabulary_json(text, top_n)
    
    # 分析結果を result.html テンプレートに渡して表示
    return render_template('result.html', words=words)

if __name__ == '__main__':
    # 環境変数からポートを取得するか、デフォルトで5000番ポートを使用
    port = int(os.environ.get("PORT", 5000))
    # Flaskアプリケーションを実行します。
    # debug=True は開発用です。本番環境では False にしてください。
    app.run(debug=True, port=port)