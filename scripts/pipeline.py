"""
LLMファインチューニング用データパイプライン
PDF抽出からデータ拡張まで一括実行

使用例:
    # 基本使用（辞書ベースの拡張のみ）
    python pipeline.py input.pdf -o output.jsonl --steps extract,dictionary,elaboration

    # フル実行（LLM拡張も含む）
    python pipeline.py input.pdf --steps all --use-azure-di

    # 既存JSONLに拡張追加
    python pipeline.py --input existing.jsonl --steps qa,english
"""

import json
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# scriptsディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# 各モジュールをインポート
from extract.datasetmaker import AdvancedPDFProcessor
from augment import expand_dictionary
from augment import expand_elaboration
from augment import expand_generalized
from augment import expand_keywords
from augment import expand_qa_difficult
from augment import expand_to_english
from augment import expand_graph_relations


# 利用可能なステップ
AVAILABLE_STEPS = [
    "extract",      # PDF抽出
    "dictionary",   # 辞書から用語定義を追加
    "elaboration",  # 用語に括弧説明を追加
    "generalized",  # 専門用語を一般表現に置換
    "keywords",     # キーワード抽出・Markdown化
    "qa",           # Q&Aペア生成 (LLM)
    "english",      # 英語翻訳 (LLM)
    "graph",        # グラフ関係性テキスト化 (LLM)
]

# LLMを使用するステップ
LLM_STEPS = ["qa", "english", "graph"]


def parse_steps(steps_str: str) -> list:
    """ステップ文字列をパース"""
    if steps_str.lower() == "all":
        return AVAILABLE_STEPS.copy()

    steps = [s.strip() for s in steps_str.split(",")]

    # 検証
    for step in steps:
        if step not in AVAILABLE_STEPS:
            raise ValueError(f"不明なステップ: {step}\n利用可能: {', '.join(AVAILABLE_STEPS)}")

    return steps


def load_jsonl(file_path: str) -> list:
    """JSONLファイルを読み込み"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: list, file_path: str):
    """JSONLファイルに保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_pdf(
    pdf_path: str,
    use_azure_di: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100
) -> list:
    """PDFを抽出してデータリストを返す"""
    print(f"\n[extract] PDF抽出: {pdf_path}")

    processor = AdvancedPDFProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extract_tables=True,
        extract_images=True,
        use_azure_di=use_azure_di
    )

    chunks = processor.process_pdf(pdf_path)

    # ChunkDataをdict形式に変換
    data = []
    for chunk in chunks:
        data.append({
            "text": chunk.text,
            "id": chunk.id,
            "source": "extract"
        })

    print(f"  抽出チャンク数: {len(data)}")
    return data


async def run_pipeline(
    input_path: Optional[str],
    output_path: str,
    steps: list,
    dict_file: str,
    graph_file: Optional[str] = None,
    use_azure_di: bool = False,
    batch_size: int = 5,
    chunk_size: int = 1500,
    chunk_overlap: int = 100
):
    """パイプラインを実行"""
    all_results = []

    # 入力データの準備
    if input_path:
        input_file = Path(input_path)

        if input_file.suffix.lower() == ".pdf":
            # PDFの場合はextractステップを実行
            if "extract" not in steps:
                steps.insert(0, "extract")

            base_data = extract_pdf(
                str(input_file),
                use_azure_di=use_azure_di,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_results.extend(base_data)

        elif input_file.suffix.lower() == ".jsonl":
            # JSONLの場合は読み込み
            print(f"\n[input] JSONL読み込み: {input_path}")
            base_data = load_jsonl(str(input_file))
            print(f"  読み込み件数: {len(base_data)}")

            # sourceが未設定の場合はextractを設定
            for item in base_data:
                if "source" not in item:
                    item["source"] = "extract"
            all_results.extend(base_data)

        else:
            raise ValueError(f"サポートされていないファイル形式: {input_file.suffix}")

    else:
        raise ValueError("入力ファイルを指定してください")

    # extractは上で処理済みなのでスキップ
    steps = [s for s in steps if s != "extract"]

    # 辞書ベースのステップを実行
    if "dictionary" in steps:
        print(f"\n[dictionary] 辞書から用語定義を生成")
        dict_data = expand_dictionary.process(dict_file)
        print(f"  生成件数: {len(dict_data)}")
        all_results.extend(dict_data)

    if "elaboration" in steps:
        print(f"\n[elaboration] 用語に括弧説明を追加")
        elab_data = expand_elaboration.process(base_data, dict_file)
        print(f"  生成件数: {len(elab_data)}")
        all_results.extend(elab_data)

    if "generalized" in steps:
        print(f"\n[generalized] 専門用語を一般表現に置換")
        gen_data = expand_generalized.process(base_data, dict_file)
        print(f"  生成件数: {len(gen_data)}")
        all_results.extend(gen_data)

    if "keywords" in steps:
        print(f"\n[keywords] キーワード抽出・Markdown化")
        kw_data = expand_keywords.process(base_data, dict_file)
        print(f"  生成件数: {len(kw_data)}")
        all_results.extend(kw_data)

    # LLMベースのステップを実行
    if "qa" in steps:
        print(f"\n[qa] Q&Aペア生成 (LLM)")
        qa_data = await expand_qa_difficult.process(base_data, dict_file, batch_size=batch_size)
        print(f"  生成件数: {len(qa_data)}")
        all_results.extend(qa_data)

    if "english" in steps:
        print(f"\n[english] 英語翻訳 (LLM)")
        en_data = await expand_to_english.process(base_data, batch_size=batch_size)
        print(f"  生成件数: {len(en_data)}")
        all_results.extend(en_data)

    if "graph" in steps:
        if graph_file and Path(graph_file).exists():
            print(f"\n[graph] グラフ関係性テキスト化 (LLM)")
            graph_data = await expand_graph_relations.process(graph_file, batch_size=batch_size)
            print(f"  生成件数: {len(graph_data)}")
            all_results.extend(graph_data)
        else:
            print(f"\n[graph] スキップ: グラフファイルが指定されていないか存在しません")

    # 結果を保存
    print(f"\n[output] 結果を保存: {output_path}")
    save_jsonl(all_results, output_path)
    print(f"  合計件数: {len(all_results)}")

    # ステップ別の内訳
    source_counts = {}
    for item in all_results:
        source = item.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\n[内訳]")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}件")


def main():
    parser = argparse.ArgumentParser(
        description="LLMファインチューニング用データパイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # PDFから辞書ベースの拡張まで実行
  python pipeline.py input.pdf -o output.jsonl --steps extract,dictionary,elaboration

  # Azure DIを使用してフル実行
  python pipeline.py input.pdf --steps all --use-azure-di

  # 既存JSONLにLLM拡張を追加
  python pipeline.py --input existing.jsonl --steps qa,english -o expanded.jsonl

利用可能なステップ:
  extract      - PDF抽出 (入力がPDFの場合は自動実行)
  dictionary   - 辞書から用語定義を追加 (無料)
  elaboration  - 用語に括弧説明を追加 (無料)
  generalized  - 専門用語を一般表現に置換 (無料)
  keywords     - キーワード抽出・Markdown化 (無料)
  qa           - Q&Aペア生成 (LLM・有料)
  english      - 英語翻訳 (LLM・有料)
  graph        - グラフ関係性テキスト化 (LLM・有料)
  all          - 全ステップ実行
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="入力ファイル (PDFまたはJSONL)"
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_alt",
        help="入力ファイル (PDFまたはJSONL) - 代替指定"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
        help="出力JSONLファイル (デフォルト: output.jsonl)"
    )
    parser.add_argument(
        "--steps", "-s",
        default="extract,dictionary,elaboration,generalized,keywords",
        help="実行するステップ (カンマ区切り or 'all')"
    )
    parser.add_argument(
        "--dict", "-d",
        default="data/dict/terms.json",
        help="専門用語辞書JSONファイル"
    )
    parser.add_argument(
        "--graph", "-g",
        default="data/graph/graph.json",
        help="ナレッジグラフJSONファイル"
    )
    parser.add_argument(
        "--use-azure-di",
        action="store_true",
        help="Azure Document Intelligenceを使用してPDFを抽出"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="LLM処理のバッチサイズ (デフォルト: 5)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="チャンクサイズ (デフォルト: 1500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="チャンクオーバーラップ (デフォルト: 100)"
    )

    args = parser.parse_args()

    # 入力ファイルの決定
    input_path = args.input or args.input_alt
    if not input_path:
        parser.error("入力ファイルを指定してください")

    # ステップをパース
    try:
        steps = parse_steps(args.steps)
    except ValueError as e:
        parser.error(str(e))

    # LLMステップがあるか確認
    has_llm_steps = any(s in LLM_STEPS for s in steps)
    if has_llm_steps:
        print("注意: LLMを使用するステップが含まれています。APIコストが発生します。")

    print(f"入力: {input_path}")
    print(f"出力: {args.output}")
    print(f"ステップ: {', '.join(steps)}")
    print(f"辞書: {args.dict}")
    if "graph" in steps:
        print(f"グラフ: {args.graph}")

    # パイプラインを実行
    asyncio.run(run_pipeline(
        input_path=input_path,
        output_path=args.output,
        steps=steps,
        dict_file=args.dict,
        graph_file=args.graph,
        use_azure_di=args.use_azure_di,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    ))

    print("\n完了!")


if __name__ == "__main__":
    main()
