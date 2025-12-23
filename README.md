# LLM継続事前学習データセット作成パイプライン

PDFから継続事前学習用のJSONLデータセットを作成し、データ拡張を行うツール群です。

## フォルダ構成

```
c:\work\finetune\
├── scripts/
│   ├── extract/
│   │   └── datasetmaker.py        # PDF→JSONL変換
│   ├── augment/
│   │   ├── expand_elaboration.py  # 専門用語に説明追加
│   │   ├── expand_generalized.py  # 専門用語→一般用語置換
│   │   ├── expand_keywords.py     # キーワード抽出・説明
│   │   ├── expand_qa_difficult.py # 専門QA生成（LLM使用）
│   │   └── expand_to_english.py   # 英語翻訳（LLM使用）
│   ├── preprocess/
│   │   ├── preprocess_dataset.py  # Dolly-JAフォーマット変換
│   │   └── clean_jsonl.py         # データ清掃
│   └── model/
│       └── merge_models.py        # モデルマージ
├── tools/
│   └── jsonl_viewer.py            # Streamlit JSONLビューアー
├── data/
│   ├── input/                     # 入力PDF
│   ├── output/                    # 出力JSONL
│   └── dict/                      # 専門用語辞書
├── config/
│   └── .env                       # 環境変数（API Key等）
└── README.md
```

## 処理フロー

```
[Phase 1: 抽出]
data/input/*.pdf
    ↓ scripts/extract/datasetmaker.py
data/output/*_preprocessed.jsonl

[Phase 2: データ拡張]
data/output/*_preprocessed.jsonl + data/dict/terms.json
    ↓ scripts/augment/expand_*.py
data/output/*_generalized.jsonl
data/output/*_elaboration.jsonl
data/output/*_keywords.jsonl
data/output/*_qa_difficult.jsonl
data/output/*_en.jsonl

[Phase 3: 清掃]
    ↓ scripts/preprocess/clean_jsonl.py
最終データセット
```

## 使い方

### 1. PDF→JSONL変換

```bash
python scripts/extract/datasetmaker.py data/input/sample.pdf --output data/output/sample.jsonl
```

### 2. データ拡張（辞書ベース、LLM不要）

```bash
# 専門用語→一般用語置換
python scripts/augment/expand_generalized.py

# 専門用語に説明追加
python scripts/augment/expand_elaboration.py

# キーワード抽出
python scripts/augment/expand_keywords.py
```

### 3. データ拡張（LLM使用）

```bash
# 事前に config/.env を設定
# AZURE_OPENAI_ENDPOINT=xxx
# AZURE_OPENAI_API_KEY=xxx

# 専門QA生成
python scripts/augment/expand_qa_difficult.py

# 英語翻訳
python scripts/augment/expand_to_english.py
```

### 4. JSONLビューアー

```bash
streamlit run tools/jsonl_viewer.py
```

## 専門用語辞書フォーマット

`data/dict/terms.json`:

```json
{
  "terms": [
    {
      "headword": "ガス軸受",
      "brief_definition": "空気で回転軸を支持する技術",
      "definition": "詳細な説明...",
      "synonyms": ["ガス潤滑軸受", "ガスベアリング"],
      "domain": "機械工学",
      "confidence": 1.0
    }
  ]
}
```

## 出力JSONL形式

```jsonl
{"text": "...", "augmentation_type": "generalized", "replacements": [...]}
{"text": "...", "augmentation_type": "elaboration", "elaborations": [...]}
{"text": "...", "augmentation_type": "keywords", "keywords": [...]}
{"text": "...", "augmentation_type": "qa_difficult", "question": "...", "answer": "..."}
```
