# app.py

from flask import Flask, request, render_template, session
from dammy import make_dic
import os

app = Flask(__name__)
app.secret_key = 'secret-key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/loading', methods=['POST'])
def loading():
    session['text'] = request.form['text']
    session['top_n'] = int(request.form['top_n'])
    return render_template('loading.html')

@app.route('/result')
def result():
    text = session.get('text', '')
    top_n = session.get('top_n', 10)

    words = make_dic(text, top_n)
    return render_template('result.html', words=words)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
