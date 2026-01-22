"""
PDFフォルダからLLM学習用データセットを一括生成するバッチパイプライン

使用例:
    # 基本（抽出のみ）
    python scripts/batch_pipeline.py data/input/ -o data/output/train.jsonl

    # データ拡張あり
    python scripts/batch_pipeline.py data/input/ -o data/output/train.jsonl --augment

    # フル装備（Azure DI + 拡張 + パッキング）
    python scripts/batch_pipeline.py data/input/ -o data/output/train.jsonl --use-azure-di --augment --pack --max-seq-len 2048
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# scriptsディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from extract.datasetmaker import AdvancedPDFProcessor
from preprocess.pack_sequences import pack_texts, estimate_tokens


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    use_azure_di: bool,
    augment: bool,
    aug_options: dict,
    clean_level: str = "basic"
) -> Optional[Path]:
    """
    単一PDFを処理してJSONLを出力

    Returns:
        出力ファイルのパス（失敗時はNone）
    """
    print(f"\n{'='*60}")
    print(f"処理中: {pdf_path.name}")
    print(f"{'='*60}")

    try:
        # PDF処理
        processor = AdvancedPDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_tables=True,
            extract_images=False,  # 画像処理は時間がかかるのでデフォルトOFF
            use_azure_di=use_azure_di,
            clean_level=clean_level
        )

        chunks = processor.process_pdf(str(pdf_path))

        if not chunks:
            print(f"  警告: チャンク抽出失敗")
            return None

        print(f"  抽出チャンク数: {len(chunks)}")

        # データ拡張（オプション）
        results = []

        # LLMベース拡張の準備
        augmenter = None
        llm_augment_enabled = any([
            aug_options.get("paraphrase"),
            aug_options.get("qa"),
            aug_options.get("summary"),
            aug_options.get("discussion"),
            aug_options.get("translation_zh"),
        ])
        if augment and llm_augment_enabled:
            from extract.datasetmaker import DataAugmenter
            augmenter = DataAugmenter()
            if not augmenter.client:
                print("  ⚠️ 警告: Azure OpenAI クライアントが初期化されていません")
                print("    → AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY を確認してください")
                print("    → LLMベースの拡張はスキップされます")

        # 辞書ベース拡張の準備
        dict_file = aug_options.get("dict_file", "data/dict/terms.json")
        term_dict = None
        if augment and (aug_options.get("generalized") or aug_options.get("keywords")):
            try:
                from augment.expand_generalized import load_dictionary
                term_dict = load_dictionary(dict_file)
                print(f"  辞書読み込み完了: {len(term_dict)}用語")
            except Exception as e:
                print(f"  ⚠️ 警告: 辞書読み込み失敗: {e}")

        for chunk in chunks:
            # オリジナル
            results.append({"text": chunk.text})

            # LLMベース拡張
            if augment and augmenter and augmenter.client:
                if aug_options.get("paraphrase"):
                    paraphrase = augmenter.generate_paraphrase(chunk.text)
                    if paraphrase:
                        results.append({"text": paraphrase})

                if aug_options.get("qa"):
                    qa = augmenter.generate_qa(chunk.text)
                    if qa and qa.get("question") and qa.get("answer"):
                        qa_text = f"質問: {qa['question']}\n回答: {qa['answer']}"
                        results.append({"text": qa_text})

                if aug_options.get("summary"):
                    summary = augmenter.generate_summary(chunk.text)
                    if summary:
                        results.append({"text": summary})

                # 議論形式
                if aug_options.get("discussion"):
                    discussion = augmenter.generate_discussion(chunk.text)
                    if discussion:
                        results.append({"text": discussion})

            # 辞書ベース拡張（LLM不要）
            if augment and term_dict:
                # 一般化
                if aug_options.get("generalized"):
                    from augment.expand_generalized import generalize_text
                    gen_result = generalize_text(chunk.text, term_dict)
                    if gen_result and gen_result.get("text") != chunk.text and gen_result.get("replacements"):
                        results.append({"text": gen_result["text"]})

                # キーワード抽出
                if aug_options.get("keywords"):
                    from augment.expand_keywords import extract_keywords, generate_keywords_text
                    kw_result = extract_keywords(chunk.text, term_dict)
                    if kw_result.get("keywords"):
                        kw_text = generate_keywords_text(kw_result["keywords"])
                        if kw_text:
                            results.append({"text": kw_text})

        # 辞書定義（チャンクとは別に辞書全体から生成）
        if augment and aug_options.get("dictionary"):
            try:
                from augment.expand_dictionary import process as dict_process
                dict_results = dict_process(dict_file)
                for item in dict_results:
                    results.append({"text": item["text"]})
                print(f"  辞書定義追加: {len(dict_results)}件")
            except Exception as e:
                print(f"  ⚠️ 警告: 辞書定義生成失敗: {e}")

        # グラフ関係性（async処理）
        if augment and aug_options.get("graph"):
            try:
                import asyncio
                from augment.expand_graph_relations import process as graph_process
                graph_file = aug_options.get("graph_file", "data/graph/graph.json")
                graph_results = asyncio.run(graph_process(graph_file))
                for item in graph_results:
                    results.append({"text": item["text"]})
                print(f"  グラフ関係性追加: {len(graph_results)}件")
            except Exception as e:
                print(f"  ⚠️ 警告: グラフ関係性生成失敗: {e}")

        # 英語翻訳（async処理）
        if augment and aug_options.get("translation_en"):
            try:
                import asyncio
                from augment.expand_to_english import process as translate_en
                data = [{"text": chunk.text} for chunk in chunks]
                translated = asyncio.run(translate_en(data))
                for item in translated:
                    results.append({"text": item["text"]})
                print(f"  英語翻訳追加: {len(translated)}件")
            except Exception as e:
                print(f"  ⚠️ 警告: 英語翻訳失敗: {e}")

        # 中国語翻訳（LLM使用）
        if augment and aug_options.get("translation_zh") and augmenter and augmenter.client:
            translated_count = 0
            for chunk in chunks:
                try:
                    translated = augmenter.generate_translation(chunk.text, "中国語")
                    if translated:
                        results.append({"text": translated})
                        translated_count += 1
                except Exception as e:
                    print(f"  ⚠️ 警告: 中国語翻訳失敗: {e}")
            if translated_count > 0:
                print(f"  中国語翻訳追加: {translated_count}件")

        print(f"  拡張後データ数: {len(results)}")

        # 出力
        output_path = output_dir / f"{pdf_path.stem}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"  出力: {output_path}")
        return output_path

    except Exception as e:
        print(f"  エラー: {e}")
        return None


def pack_jsonl(
    input_path: Path,
    output_path: Path,
    max_seq_len: int,
    separator: str = "\n\n"
) -> int:
    """
    JSONLをパッキングして出力

    Returns:
        パッキング後のデータ数
    """
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("text"):
                    texts.append(data["text"])
            except json.JSONDecodeError:
                continue

    if not texts:
        return 0

    packed = pack_texts(texts, max_seq_len, separator)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in packed:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    return len(packed)


def merge_jsonl_files(input_files: list[Path], output_path: Path, shuffle: bool = True, tokenizer_name: str = None):
    """
    複数のJSONLファイルをマージ

    Args:
        tokenizer_name: トークナイザーのモデル名（指定時は正確なトークン数を計算）

    Returns:
        dict: {"count": int, "total_chars": int, "total_tokens": int, "is_estimated": bool}
    """
    all_records = []

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    all_records.append(record)
                except json.JSONDecodeError:
                    continue

    if shuffle:
        import random
        random.shuffle(all_records)

    # 統計計算
    total_chars = 0
    all_texts = []
    for record in all_records:
        if "text" in record:
            total_chars += len(record["text"])
            all_texts.append(record["text"])

    # トークン数計算
    is_estimated = True
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            print(f"トークナイザー読み込み中: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            total_tokens = 0
            for text in all_texts:
                total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            is_estimated = False
            print(f"トークン数計算完了（正確）")
        except Exception as e:
            print(f"トークナイザー読み込み失敗: {e}")
            print("推定値を使用します")
            total_tokens = int(total_chars * 0.5)
    else:
        # 日本語は1文字≒0.5トークン程度（目安）
        total_tokens = int(total_chars * 0.5)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return {
        "count": len(all_records),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "is_estimated": is_estimated
    }


def main():
    parser = argparse.ArgumentParser(
        description="PDFフォルダからLLM学習用データセットを一括生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 入出力
    parser.add_argument("input", help="入力PDFフォルダ または 入力JSONLファイル")
    parser.add_argument("-o", "--output", required=True, help="最終出力ファイル（train.jsonl）")

    # PDF処理オプション
    parser.add_argument("--chunk-size", type=int, default=1500, help="チャンクサイズ")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="チャンクオーバーラップ")
    parser.add_argument("--use-azure-di", action="store_true", help="Azure Document Intelligenceを使用")

    # データ拡張オプション
    parser.add_argument("--augment", action="store_true", help="データ拡張を有効化")
    parser.add_argument("--aug-paraphrase", action="store_true", help="言い換え生成")
    parser.add_argument("--aug-qa", action="store_true", help="Q&A生成")
    parser.add_argument("--aug-summary", action="store_true", help="要約生成")
    parser.add_argument("--aug-keywords", action="store_true", help="キーワード抽出")
    parser.add_argument("--aug-translation-en", action="store_true", help="英語翻訳")
    parser.add_argument("--aug-translation-zh", action="store_true", help="中国語翻訳")
    parser.add_argument("--aug-discussion", action="store_true", help="議論形式")

    # 辞書/グラフベース拡張
    parser.add_argument("--aug-dictionary", action="store_true", help="辞書から用語定義 (LLM不要)")
    parser.add_argument("--aug-generalized", action="store_true", help="専門用語→一般用語 (LLM不要)")
    parser.add_argument("--aug-graph", action="store_true", help="グラフ関係性 (LLM使用)")
    parser.add_argument("--dict-file", default="data/dict/terms.json", help="辞書ファイルパス")
    parser.add_argument("--graph-file", default="data/graph/graph.json", help="グラフファイルパス")

    # パッキングオプション
    parser.add_argument("--pack", action="store_true", help="パッキングを有効化")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="最大シーケンス長")

    # クリーニングオプション
    parser.add_argument("--clean-level", choices=["off", "basic", "aggressive"], default="basic",
                        help="テキストクリーニングレベル (off=なし, basic=基本, aggressive=積極的)")
    parser.add_argument("--no-toc-removal", action="store_true", help="目次除去を無効化")

    # その他
    parser.add_argument("--no-shuffle", action="store_true", help="最終マージ時にシャッフルしない")
    parser.add_argument("--keep-intermediate", action="store_true", help="中間ファイルを保持")
    parser.add_argument("--tokenizer", help="トークナイザーのモデル名（例: llm-jp/llm-jp-3-13b）")

    args = parser.parse_args()

    # パス設定
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 拡張オプション
    aug_options = {
        "paraphrase": args.aug_paraphrase or args.augment,
        "qa": args.aug_qa or args.augment,
        "summary": args.aug_summary,
        "keywords": args.aug_keywords,
        "translation_en": args.aug_translation_en,
        "translation_zh": args.aug_translation_zh,
        "discussion": args.aug_discussion,
        "dictionary": args.aug_dictionary,
        "generalized": args.aug_generalized,
        "graph": args.aug_graph,
        "dict_file": args.dict_file,
        "graph_file": args.graph_file,
    }

    # 入力タイプ判定: JSONLファイル or PDFフォルダ
    is_jsonl_input = input_path.is_file() and input_path.suffix.lower() in ['.jsonl', '.json']

    if is_jsonl_input:
        # JSONL入力モード（拡張のみ）
        print(f"\n{'#'*60}")
        print(f"# バッチパイプライン開始 (JSONL入力モード)")
        print(f"# 入力ファイル: {input_path}")
        print(f"# 出力先: {output_path}")
        print(f"{'#'*60}")

        processed_files = [input_path]
    else:
        # PDFフォルダ入力モード
        input_dir = input_path
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"エラー: {input_dir} にPDFファイルがありません")
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"# バッチパイプライン開始 (PDFフォルダモード)")
        print(f"# 入力フォルダ: {input_dir}")
        print(f"# PDFファイル数: {len(pdf_files)}")
        print(f"# 出力先: {output_path}")
        print(f"{'#'*60}")

        # Phase 1: PDF処理
        print(f"\n[Phase 1] PDF処理")
        processed_files = []
        for pdf_path in pdf_files:
            result = process_single_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                use_azure_di=args.use_azure_di,
                augment=args.augment,
                aug_options=aug_options,
                clean_level=args.clean_level
            )
            if result:
                processed_files.append(result)

    if not processed_files:
        print("エラー: 処理されたファイルがありません")
        sys.exit(1)

    if not is_jsonl_input:
        print(f"\n処理完了: {len(processed_files)}/{len(pdf_files)} ファイル")

    # Phase 2: パッキング（オプション）
    files_to_merge = processed_files
    if args.pack:
        print(f"\n[Phase 2] パッキング (max_seq_len={args.max_seq_len})")
        packed_files = []
        for jsonl_path in processed_files:
            packed_path = jsonl_path.with_name(f"{jsonl_path.stem}_packed.jsonl")
            count = pack_jsonl(jsonl_path, packed_path, args.max_seq_len)
            print(f"  {jsonl_path.name} → {packed_path.name} ({count}件)")
            packed_files.append(packed_path)
        files_to_merge = packed_files

    # Phase 3: マージ
    print(f"\n[Phase 3] マージ")
    stats = merge_jsonl_files(
        files_to_merge,
        output_path,
        shuffle=not args.no_shuffle,
        tokenizer_name=args.tokenizer
    )

    # 中間ファイル削除
    if not args.keep_intermediate:
        for f in processed_files:
            if f.exists() and f != output_path:
                f.unlink()
        if args.pack:
            for f in packed_files:
                if f.exists() and f != output_path:
                    f.unlink()

    # 完了
    token_label = "トークン数" if not stats['is_estimated'] else "推定トークン数"
    print(f"\n{'#'*60}")
    print(f"# 完了!")
    print(f"# 出力ファイル: {output_path}")
    print(f"# 総データ数: {stats['count']}件")
    print(f"# 合計文字数: {stats['total_chars']:,}文字")
    print(f"# {token_label}: {stats['total_tokens']:,}トークン")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
