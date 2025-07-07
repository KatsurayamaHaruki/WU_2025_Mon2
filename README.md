# 英語テキスト TF-IDF 分析アプリケーション

## 概要

このWebアプリケーションは、入力された英語のテキストを自然言語処理技術を用いて分析し、重要単語を抽出して一覧表示するツールです。
[TF-IDF](https://ja.wikipedia.org/wiki/Tf-idf)スコアに基づいてテキスト内での単語の重要度を算出し、それぞれの単語について**日本語訳**、**CEFRレベル**、**品詞**を提供します。

## 機能

* **重要単語の抽出**: TF-IDFアルゴリズムを利用して、入力されたテキスト内で特徴的・重要と思われる単語を抽出します。
* **単語情報の付与**: 抽出した単語に対し、以下の情報を付与して表示します。
    * **日本語訳**: `googletrans`ライブラリを用いて単語の日本語訳を自動で取得します。
    * **CEFRレベル**: CEFR-JおよびOCTANOVEの語彙リストに基づき、各単語の言語習熟度レベル（A1〜C2）を判定します。
    * **品詞**: `spaCy`を用いて各単語の品詞（名詞、動詞など）を特定し、日本語で表示します。
* **Webインターフェース**: シンプルなWeb画面から、誰でも簡単にテキスト分析を実行できます。

## 技術スタック

* **バックエンド**: Flask
* **自然言語処理**: spaCy, scikit-learn, pandas
* **翻訳**: googletrans
* **フロントエンド**: HTML, JavaScript

## セットアップと実行方法

### 1. 前提条件

* Python 3.9 以降がインストールされていること。

### 2. 必要なライブラリのインストール

以下のコマンドを実行して、必要なPythonライブラリをすべてインストールします。

```bash
pip install Flask spacy scikit-learn pandas googletrans==4.0.0-rc1 httpx legacy-cgi
```
次に、自然言語処理ライブラリspaCyの英語モデルをダウンロードします。

```
python -m spacy download en_core_web_sm
```
アプリ起動
```
pip install flask
python app.py
```
