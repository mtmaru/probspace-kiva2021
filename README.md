# ProbSpace クラファンコンペ 2nd Place Solution

ProbSpace[「Kiva／クラウドファンディングの資金調達額予測」](https://comp.probspace.com/competitions/kiva2021)の 2nd place solution です。

## パイプライン

全モデルに共通するパイプラインは以下の通りです。<br />
個々のモデルの詳細は、[コード > モデル作成](#モデル作成)を参照してください。

![pipeline](/pipeline.png)

## コード

### モデル作成

詳しくは[実験条件の詳細](#実験条件の詳細)を参照してください。<br />
最後の `maruyama_20220213_03.ipynb` が、すべてのモデルをアンサンブルした最終提出版です。

|ファイル名|テキストの前処理|テキストの最大長|テキストのモデル|カスタムヘッダー|画像のモデル|embedding_dim|追加の特徴量|Dence層の数|重み付け|オプティマイザー|エポック数|ホールドアウト|Public|Private|
|:---|:---|---:|:---|:---|:---|---:|:---|---:|:---|:---|---:|---:|---:|---:|
|maruyama_20220101_01.py|タグ等を除去、小文字変換|128|roberta-base|CLSのみ|ResNet152|8|(なし)|3|(なし)|Adam|10|218.70|269.4038|266.3941|
|maruyama_20220103_01.py|タグ等を除去、小文字変換|512|roberta-base|CLSのみ|ResNet152|8|(なし)|3|(なし)|Adam|10|225.50|273.9649|271.5372|
|maruyama_20220104_01.py|タグ等を除去、小文字変換|512|roberta-base|CLSのみ|ResNet152|16|(なし)|3|(なし)|Adam|10|220.71|(未投稿)|(未投稿)|
|maruyama_20220105_01.py|タグ等を除去|512|roberta-base|CLSのみ|ResNet152|8|(なし)|3|(なし)|Adam|10|241.58|(未投稿)|(未投稿)|
|maruyama_20220108_02.py|タグ等を除去、小文字変換|512|roberta-base|CLSのみ|(なし)|8|人数|3|(なし)|Adam|10|216.91|265.8470|260.7173|
|maruyama_20220116_01.py|タグ等を除去、小文字変換|512|roberta-base|Conv1d|(なし)|8|人数|3|(なし)|Adam|10|221.49|271.2945|267.9450|
|maruyama_20220116_02.py|タグ等を除去、小文字変換|512|roberta-base|Conv1d|(なし)|8|人数|3|(なし)|Adam|20|212.39|(未投稿)|(未投稿)|
|maruyama_20220118_01.py|タグ等を除去、小文字変換|512|roberta-base|CLSのみ|(なし)|8|希望金額|3|(なし)|Adam|10|236.39|(未投稿)|(未投稿)|
|maruyama_20220123_01.py|(なし)|512|roberta-base|CLSのみ|(なし)|8|人数|2|(なし)|AdamW|10|220.82|267.7851|264.1726|
|maruyama_20220205_01.py|(なし)|512|roberta-base|CLSのみ|(なし)|8|人数|2|密度比|AdamW|10|231.28|277.1100|270.7795|
|maruyama_20220213_01.py|(なし)|512|deberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|8|225.95|(未投稿)|(未投稿)|
|maruyama_20220213_01.py|(なし)|512|deberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|9|228.79|(未投稿)|(未投稿)|
|maruyama_20220213_01.py|(なし)|512|deberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|10|234.98|(未投稿)|(未投稿)|
|maruyama_20220213_02.py|(なし)|512|roberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|8|230.56|(未投稿)|(未投稿)|
|maruyama_20220213_02.py|(なし)|512|roberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|9|228.41|(未投稿)|(未投稿)|
|maruyama_20220213_02.py|(なし)|512|roberta-base|CLSのみ|(なし)|8|(なし)|2|(なし)|AdamW|10|220.62|(未投稿)|(未投稿)|
|maruyama_20220213_03.ipynb|-|-|-|-|-|-|-|-|-|-|-|(未検証)|248.3038|244.7570|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|

### 特徴抽出

- 画像から人数を抽出
    - `count_num_faces.py`
- DESCRIPTION_TRANSLATEDから希望金額を抽出
    - `extract_loan_amount_local.py`

## 実験条件の詳細

いろいろ試しましたが、どれも精度はあまり変わりませんでした。<br />
精度に差がない場合はシンプルな方を採用する、という方針でモデルを選びました。

### 特徴抽出器

- テキストの前処理
    - HTMLタグ等の削除と小文字変換を試しました。
    - あまり精度が変わらなかったため、前処理はせず、そのままBERTへ入れることにしました。
- テキストの最大長
    - 128単語と512単語を試しました。
    - あまり精度が変わらなかったため、情報の損失が少なくて済む512単語を採用しました。
    - 512単語にするにあたり、tokenizeのタイミングをデータセット作成時から推論時に移しました。
- テキストの学習済みモデル
    - RoBERTaとDeBERTaを試しました。
    - RoBERTaのほうが精度が良かったため、RoBERTaを採用しました。
- BERTのカスタムヘッダー
    - CLSのみ使うパターンと全隠れ状態をConv1dで畳み込むパターンを試しました。
    - あまり精度が変わらなかったため、シンプルなCLSのみを採用しました。
- 画像の学習済みモデル
    - ResNet152を試しました。
    - あまり精度が変わらなかったため、画像は使わないことにしました。
- カテゴリー変数のembedding_dim
    - 8次元と16次元を試しました。
    - あまり精度が変わらなかったため、シンプルな8次元を採用しました。
- 追加の特徴量
    - 画像から抽出した人数と説明から抽出した希望金額を特徴量に追加しました。
    - どちらもあまり精度が変わらなかったため、追加しないことにしました。

### 予測器

- Dence層の数
    - 3層と2層を試しました。
    - どちらもあまり精度が変わらなかったため、シンプルな2層を採用しました。

### 学習

- ファインチューニング
    - BERTとResNetは、最終層のみ重みを更新し、それ以外の層の重みは固定しました。
- 損失関数
    - 対数変換＆標準化した融資額に対してMSELossを使いました。
- 損失関数の重み付け
    - 学習期間とテスト期間が離れていたことからデータセットシフトが問題になると考え、[損失関数を密度比で重み付け](https://arxiv.org/abs/2006.04662)しました。
    - あまり精度が変わらなかったため、重み付けはしないことにしました。
- オプティマイザー
    - AdamとAdamWを試しました。
    - あまり精度が変わらなかったため、Kaggleでよく使われているAdamWを採用しました。
- エポック数
    - Early stoppingは行わず、dropoutをきつめにして学習が収束するまで回す方針にしました。
    - どのモデルも8エポック目あたりで検証データに対する精度が収束しました。

### アンサンブル

- 各モデルの予測結果の加算平均を取りました。

### 後処理

- 説明に希望金額が書かれている場合、予測結果をドル換算した希望金額に置き換えました。
- 為替レートには、学習データの融資額 (ドル) を希望金額 (現地通貨) で割ったものを使いました。

### 精度検証

- ホールドアウト法で精度を評価しました。
- CVにしなかった理由は、時間がかかるからと、データの選び方に依らず精度が安定していたからです。
