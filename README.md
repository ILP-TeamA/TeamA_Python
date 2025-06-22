# TeamA_Python
🍺 ビール売上予測システム

概要
このシステムは、過去の売上実績と気象データを活用した機械学習により、ビールの日次売上量を予測し、最適な発注量を提案するAIシステムです。

📁 プロジェクト構成

beer-prediction-system/
├── model_training/              # モデル訓練関連
│   ├── model_train.py          # メインの訓練スクリプト
│   └── data/                   # 訓練データ
│       ├── sales.csv           # 売上実績データ（2024/4/1～2025/3/31）
│       └── weather.csv         # 気象データ（同期間）
├── trained_models_hyperopt/     # 訓練済みモデル保存先
│   ├── metadata.pkl            # モデルメタデータ
│   ├── scalers.pkl            # データ正規化器
│   └── {beer_name}_model.pkl  # 各ビール種類のモデル
├── function_app.py             # 予測API（Azure Functions）
└── README.md                   # 本ドキュメント

🤖 モデル訓練

データ期間

2024年4月1日 ～ 2025年3月31日 の1年間の実績データを使用

訓練内容

売上実績データ: 各ビール種類の日次売上数量
気象データ: 気温、降水量、日照時間、湿度等
特徴工程: 曜日、季節性、顧客数、天気要因等の複合特徴量生成
ハイパーパラメータ最適化: 最適なモデルパラメータの自動探索

対象ビール種類

ペールエール (Pale Ale)

ラガー (Lager)

IPA (India Pale Ale)

ホワイトビール (White Beer)

黒ビール (Dark Beer)

フルーツビール (Fruit Beer)

🚀 使用方法

1. モデル訓練の実行
cd model_training
python model_train.py

実行結果:

訓練済みモデルが trained_models_hyperopt/ フォルダに保存されます
各ビール種類ごとに最適化されたモデルファイルが生成されます

2. 予測システムの起動

Azure Functions環境で実行
func start

または直接Python実行
python function_app.py

3. 予測API呼び出し

POST リクエスト例

curl -X POST "https://your-function-url/api/predictor" \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2025-04-15"}'


レスポンス例

json{
  "type": "monday_order_recommendations",
  "prediction_date": "2025-04-15",
  "day_of_week": "周一",
  "beer_purchase_recommendations": {
    "ペールエール": 32.4,
    "ラガー": 27.8,
    "IPA": 18.2,
    "ホワイトビール": 13.6,
    "黒ビール": 7.5,
    "フルーツビール": 5.8
  },
  "total_order_amount": 105.3
}

📊 システム機能

🎯 スマート発注提案

月曜日: 火曜～木曜の3日分の発注量を提案
木曜日: 金曜～土曜＋翌週月曜の3日分の発注量を提案
その他の曜日: 「このサービスは月曜日と木曜日のみ利用可能」を表示

🌤️ 気象データ連携

OpenWeather API との連携
気温、降水量、天候が売上予測に反映
過去データ、現在データ、予報データの自動取得

🔄 インテリジェント予測

機械学習モデル: 各ビール種類に最適化された予測アルゴリズム
特徴工程: 顧客数、曜日効果、季節性、天気影響等を総合分析
成長率考慮: 前年同期比成長率を予測に反映

⚙️ 技術仕様

開発環境

Python 3.11+

Azure Functions (サーバーレス実行環境)

機械学習ライブラリ: scikit-learn, pandas, numpy

気象API: OpenWeather API

データベース: PostgreSQL (Azure Database)

主要ライブラリ

pythonazure-functions

pandas

numpy

scikit-learn==1.3.2

sqlalchemy

requests

📈 予測精度

性能指標

平均予測精度: 66.9%以上

