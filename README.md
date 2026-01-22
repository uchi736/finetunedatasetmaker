# LLM継続事前学習データセット作成パイプライン

PDFから継続事前学習用のJSONLデータセットを作成し、データ拡張を行うツール群です。

## フォルダ構成

```
c:\work\finetune\
├── scripts/
│   ├── pipeline.py                   # 統合パイプライン（一括実行）
│   ├── extract/
│   │   └── datasetmaker.py           # PDF→JSONL変換
│   ├── augment/
│   │   ├── expand_dictionary.py      # 辞書から用語定義テキスト生成
│   │   ├── expand_elaboration.py     # 専門用語に説明追加
│   │   ├── expand_generalized.py     # 専門用語→一般用語置換
│   │   ├── expand_keywords.py        # キーワード抽出・説明
│   │   ├── expand_qa_difficult.py    # 専門QA生成（LLM使用）
│   │   ├── expand_to_english.py      # 英語翻訳（LLM使用）
│   │   └── expand_graph_relations.py # ナレッジグラフ関係性テキスト化（LLM使用）
│   └── preprocess/
│       ├── preprocess_dataset.py     # Dolly-JAフォーマット変換
│       └── azure_di_processor.py     # Azure Document Intelligence処理
├── tools/
│   └── jsonl_viewer.py               # Streamlit JSONLビューアー
├── data/
│   ├── input/                        # 入力PDF
│   ├── output/                       # 出力JSONL
│   └── dict/                         # 専門用語辞書
├── config/
│   └── .env                          # 環境変数（API Key等）
├── graph.json                        # ナレッジグラフ（用語間関係）
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
data/output/*_dictionary.jsonl      # 用語定義
data/output/*_elaboration.jsonl     # 説明追加
data/output/*_generalized.jsonl     # 一般化
data/output/*_keywords.jsonl        # キーワード
data/output/*_qa_difficult.jsonl    # Q&A
data/output/*_en.jsonl              # 英語翻訳

graph.json
    ↓ scripts/augment/expand_graph_relations.py
data/output/graph_relations.jsonl   # 用語間関係テキスト

[Phase 3: 統合・清掃]
    ↓ scripts/preprocess/*.py
最終データセット
```

## 使い方

### 環境変数の設定

`config/.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_DI_ENDPOINT=https://your-di-endpoint.cognitiveservices.azure.com
AZURE_DI_API_KEY=your-di-api-key
```

### 1. PDF→JSONL変換（datasetmaker.py）

```powershell
# 基本（チャンク分割のみ）
python scripts/extract/datasetmaker.py data/input/sample.pdf --output data/output/sample.jsonl

# 高品質抽出（Azure Document Intelligence使用）
python scripts/extract/datasetmaker.py data/input/sample.pdf --output data/output/sample.jsonl --use-azure-di

# LLM学習用（データ拡張あり）
python scripts/extract/datasetmaker.py data/input/sample.pdf --output data/output/sample.jsonl --use-azure-di --enable-augmentation --aug-paraphrase --aug-qa

# フル装備
python scripts/extract/datasetmaker.py data/input/sample.pdf --output data/output/sample.jsonl --chunk-size 2000 --use-azure-di --enable-quality-control --enable-augmentation --aug-paraphrase --aug-qa --aug-summary --aug-elaboration
```

**オプション一覧**:

| オプション | 説明 |
|-----------|------|
| `--chunk-size` | チャンクサイズ（デフォルト: 1500） |
| `--chunk-overlap` | オーバーラップ（デフォルト: 100） |
| `--use-azure-di` | Azure Document Intelligence使用 |
| `--no-extract-images` | 画像抽出を無効化 |
| `--no-extract-tables` | 表抽出を無効化 |
| `--enable-quality-control` | 品質管理（重複・低品質除去） |
| `--enable-augmentation` | データ拡張を有効化 |
| `--aug-paraphrase` | 言い換え生成 |
| `--aug-qa` | Q&A生成 |
| `--aug-summary` | 要約生成 |
| `--aug-keywords` | キーワード抽出 |
| `--aug-elaboration` | 詳細化 |
| `--aug-translation-en` | 英語翻訳 |
| `--aug-translation-zh` | 中国語翻訳 |
| `--aug-discussion` | 議論形式生成 |

### 2. 統合パイプライン（pipeline.py）

```powershell
# 基本使用（辞書ベースの拡張のみ）
python scripts/pipeline.py data/input/sample.pdf -o output.jsonl --steps extract,dictionary,elaboration

# フル実行（LLM拡張も含む）
python scripts/pipeline.py data/input/sample.pdf --steps all --use-azure-di

# 既存JSONLに拡張追加
python scripts/pipeline.py --input existing.jsonl --steps qa,english
```

**利用可能なステップ**:
- `extract` - PDF抽出
- `dictionary` - 辞書から用語定義追加
- `elaboration` - 用語に括弧説明追加
- `generalized` - 専門用語を一般表現に置換
- `keywords` - キーワード抽出・Markdown化
- `qa` - Q&Aペア生成（LLM）
- `english` - 英語翻訳（LLM）

### 3. 個別スクリプト実行

**辞書ベース（LLM不要）**:
```powershell
python scripts/augment/expand_dictionary.py
python scripts/augment/expand_elaboration.py
python scripts/augment/expand_generalized.py
python scripts/augment/expand_keywords.py
```

**LLM使用**:
```powershell
python scripts/augment/expand_qa_difficult.py
python scripts/augment/expand_to_english.py
python scripts/augment/expand_graph_relations.py
```

### 4. ナレッジグラフ関係性テキスト化

`graph.json`から用語間の関係を自然な日本語テキストに変換：

```powershell
python scripts/augment/expand_graph_relations.py
```

**出力例**:
```json
{"text": "ガス軸受は軸受の一種で、軸支持や油潤滑軸受と関連し、電動ターボ機械や輸送機器に適用される。", "node": "ガス軸受", "relation_count": 15}
```

### 5. JSONLビューアー

```powershell
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

## ナレッジグラフフォーマット

`graph.json`:

```json
{
  "graph": {
    "directed": true,
    "multigraph": true,
    "nodes": [{"id": "ガス軸受"}, {"id": "電動ターボ機械"}],
    "edges": [
      {"type": "APPLIES_TO", "source": "ガス軸受", "target": "電動ターボ機械", "key": 0},
      {"type": "IS_A", "source": "ガス軸受", "target": "軸受", "key": 0}
    ]
  }
}
```

**エッジタイプ**:
- `RELATED_TO` - 関連
- `APPLIES_TO` - 適用
- `IS_A` - 分類
- `AFFECTS` - 影響
- `HAS_ATTRIBUTE` - 属性
- `PART_OF` - 構成要素
- `BELONGS_TO_CATEGORY` - カテゴリ

## 出力JSONL形式

```jsonl
{"text": "...", "augmentation_type": "dictionary"}
{"text": "...", "augmentation_type": "elaboration", "elaborations": [...]}
{"text": "...", "augmentation_type": "generalized", "replacements": [...]}
{"text": "...", "augmentation_type": "keywords", "keywords": [...]}
{"text": "...", "augmentation_type": "qa_difficult", "question": "...", "answer": "..."}
{"text": "...", "augmentation_type": "graph_relations", "node": "...", "relation_count": 5}
```
