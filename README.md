# 🍺 ビール売上予測システム

## 📋 目次

- [概要](#-概要)
- [システム構成](#-システム構成)
- [データセット](#-データセット)
- [セットアップ](#-セットアップ)
- [使用方法](#-使用方法)
- [API仕様](#-api仕様)
- [予測結果例](#-予測結果例)
- [技術仕様](#-技術仕様)
- [性能指標](#-性能指標)

---

## 🎯 概要

このシステムは、**2024年4月1日～2025年3月31日**の1年間の売上実績と気象データを活用した機械学習により、ビールの日次売上量を予測し、最適な発注量を提案するAIシステムです。


## 📁 システム構成

```
beer-prediction-system/
├── 📁 model_training/              # モデル訓練関連
│   ├── 🐍 model_train.py          # メイン訓練スクリプト
│   └── 📁 data/                   # 訓練データ
│       ├── 📊 sales.csv           # 売上実績データ（2024/4/1～2025/3/31）
│       └── 🌤️ weather.csv         # 気象データ（同期間）
├── 📁 trained_models_hyperopt/     # 訓練済みモデル保存先
│   ├── 📋 metadata.pkl            # モデルメタデータ
│   ├── ⚖️ scalers.pkl             # データ正規化器
│   └── 🤖 {beer_name}_model.pkl   # 各ビール種類のモデル
├── ⚙️ function_app.py             # 予測API（Azure Functions）
├── 📋 requirements.txt            # 依存関係
└── 📖 README.md                   # 本ドキュメント
```

---

## 📊 データセット

### 📈 売上データ (`sales.csv`)

<details>
<summary>📋 データ形式詳細</summary>

```csv
日付,曜日,来客数,ペールエール(本),ラガー(本),IPA(本),ホワイトビール(本),黒ビール(本),フルーツビール(本)
2024-04-01,月,25,8,7,5,4,2,1
2024-04-02,火,30,12,9,7,5,3,2
2024-04-03,水,28,10,8,6,4,2,1
...
```

**含まれるデータ:**
- 📅 日付・曜日情報
- 👥 来客数データ  
- 🍺 各ビール種類別売上数量
- 💰 売上金額データ

</details>

### 🌦️ 気象データ (`weather.csv`)

<details>
<summary>📋 データ形式詳細</summary>

```csv
日付,平均気温(℃),最高気温(℃),最低気温(℃),降水量(mm),日照時間(時間),湿度(%)
2024-04-01,18.5,23.2,14.1,0.0,8.5,65
2024-04-02,19.1,24.8,15.2,2.3,6.2,72
2024-04-03,17.8,22.5,13.9,5.1,4.8,78
...
```

**含まれるデータ:**
- 🌡️ 気温情報（平均・最高・最低）
- 🌧️ 降水量データ
- ☀️ 日照時間
- 💧 湿度情報

</details>

### 🍺 対象ビール種類

| ビール種類 | 英語名 |
|------------|---------|
| 🍺 ペールエール | Pale Ale |
| 🍻 ラガー | Lager |
| 🍺 IPA | India Pale Ale |
| 🍺 ホワイトビール | White Beer |
| 🖤 黒ビール | Dark Beer |
| 🍓 フルーツビール | Fruit Beer |

---

## 🚀 セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/beer-prediction-system.git
cd beer-prediction-system
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

```bash
# OpenWeather API キーの設定
export OPENWEATHER_API_KEY="your_api_key_here"

# Azure Database接続情報（オプション）
export AZURE_DB_HOST="your_host"
export AZURE_DB_PASSWORD="your_password"
```

---

## 🔧 使用方法

### ステップ 1: モデル訓練

```bash
cd model_training
python model_train.py
```

**実行結果:**
- ✅ 訓練済みモデルが `trained_models_hyperopt/` に保存
- ✅ 各ビール種類に最適化されたモデル生成
- ✅ ハイパーパラメータ最適化完了

### ステップ 2: 予測APIの起動

#### Azure Functions環境

```bash
func start
```

#### ローカル実行

```bash
python function_app.py
```

### ステップ 3: 予測の実行

#### cURLでのAPI呼び出し

```bash
curl -X POST "https://your-function-url/api/predictor" \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2025-04-15"}'
```

#### Pythonでの呼び出し例

```python
import requests
import json

url = "https://your-function-url/api/predictor"
data = {"target_date": "2025-04-15"}

response = requests.post(url, json=data)
result = response.json()

print(f"発注推奨量: {result['beer_purchase_recommendations']}")
```

---

## 📡 API仕様

### エンドポイント

```
POST https://your-function-url/api/predictor
```

### リクエスト形式

```json
{
  "target_date": "2025-04-15"
}
```

### レスポンス形式

#### 🟢 週一（月曜日）の場合

```json
{
  "type": "monday_order_recommendations",
  "base_date": "2025-04-15",
  "order_period": "周二至周四 (3天)",
  "beer_purchase_recommendations": {
    "ペールエール": 32.4,
    "ラガー": 27.8,
    "IPA": 18.2,
    "ホワイトビール": 13.6,
    "黒ビール": 7.5,
    "フルーツビール": 5.8
  },
  "total_order_amount": 105.3,
  "daily_predictions": {
    "2025-04-16": { "date": "2025-04-16", "day_of_week": "周二", ... },
    "2025-04-17": { "date": "2025-04-17", "day_of_week": "周三", ... },
    "2025-04-18": { "date": "2025-04-18", "day_of_week": "周四", ... }
  }
}
```

#### 🟡 木曜日の場合

```json
{
  "type": "thursday_order_recommendations",
  "base_date": "2025-04-18",
  "order_period": "周五至周六及下周一 (3天)",
  "beer_purchase_recommendations": {
    "ペールエール": 35.7,
    "ラガー": 30.2,
    // ... 其他ビール
  },
  "total_order_amount": 114.9
}
```

#### 🔴 サービス利用不可の場合

```json
{
  "message": "該服務僅在周一與周四使用",
  "target_date": "2025-04-16",
  "day_of_week": "周二",
  "status": "service_unavailable",
  "available_days": ["周一", "周四"],
  "beer_purchase_recommendations": {}
}
```

---

## 📊 予測結果例

### 💼 実際の予測シナリオ

#### シナリオ 1: 春の平日（4月15日 月曜日）

**条件:**
- 🌡️ 気温: 22°C (快適)
- ☀️ 天候: 晴れ
- 👥 予想来客数: 28人

**予測結果:**

| ビール種類 | 予測販売数 | 発注推奨数 |
|------------|------------|------------|
| ペールエール | 27本 | **32本** |
| ラガー | 23本 | **28本** |
| IPA | 15本 | **18本** |
| ホワイト | 11本 | **14本** |
| 黒ビール | 6本 | **8本** |
| フルーツ | 5本 | **6本** |

**📦 合計発注推奨: 106本** （3日分）

---

## ⚙️ 技術仕様

### 🐍 開発環境

| 項目 | バージョン |
|------|------------|
| Python | 3.8+ |
| Azure Functions | Core Tools 4.x |
| scikit-learn | 1.3+ |
| pandas | 1.5+ |
| numpy | 1.24+ |

### 🧠 機械学習モデル

- **アルゴリズム**: RandomForest / ExtraTrees / GradientBoosting（ハイパーパラメータ最適化済み）
- **特徴量**: 62種類（時系列・気象・顧客数・交互作用項）
- **訓練期間**: 2024/4/1～2025/3/31（365日間）
- **検証方法**: クロスバリデーション

### ☁️ インフラ

- **Azure Functions**: サーバーレス実行環境
- **Azure Database**: PostgreSQL データストレージ
- **OpenWeather API**: リアルタイム気象データ
- **GitHub Actions**: CI/CD パイプライン

---

## 📈 性能指標

### 🎯 予測精度

| 指標 | 値 |
|------|----|
| **平均R²スコア** | 0.669 |

---

