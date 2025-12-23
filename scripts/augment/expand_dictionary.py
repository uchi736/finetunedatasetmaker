"""
専門用語辞書から用語と説明を抽出してJSONLに追加するスクリプト
辞書の各用語を学習データとして出力する
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def extract_dictionary_entries(dict_file: str) -> list:
    """
    専門用語辞書から用語エントリを抽出

    Returns:
        list: [{"term": str, "definition": str}, ...]
    """
    with open(dict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    seen_headwords = set()

    for term in data.get("terms", []):
        headword = term.get("headword", "")
        if not headword or headword in seen_headwords:
            continue

        seen_headwords.add(headword)

        # brief_definitionとdefinitionを組み合わせ
        brief_def = term.get("brief_definition", "")
        full_def = term.get("definition", "")

        # 定義文を構築
        if brief_def and full_def:
            definition = f"{brief_def}。{full_def}"
        elif brief_def:
            definition = brief_def
        elif full_def:
            definition = full_def
        else:
            continue  # 定義がない場合はスキップ

        entries.append({
            "term": headword,
            "definition": definition.strip()
        })

    return entries


def generate_training_texts(entries: list, format_type: str = "all") -> list:
    """
    用語エントリから学習用テキストを生成

    Args:
        entries: 用語エントリのリスト
        format_type: 出力形式
            - "all": 全フォーマットを1つのテキストに含める（デフォルト）
            - "definition": 「用語とは、定義である。」
            - "simple": 「用語: 定義」
            - "qa": 「Q: 用語とは何か？ A: 定義」
    """
    texts = []

    for entry in entries:
        term = entry["term"]
        definition = entry["definition"]

        if format_type == "all":
            # 全フォーマットを1つのテキストにまとめる
            text = f"""{term}: {definition}

{term}とは、{definition}

Q: {term}とは何ですか？
A: {definition}"""
        elif format_type == "definition":
            text = f"{term}とは、{definition}"
        elif format_type == "simple":
            text = f"{term}: {definition}"
        elif format_type == "qa":
            text = f"Q: {term}とは何ですか？\nA: {definition}"
        else:
            text = f"{term}: {definition}"

        texts.append({"text": text})

    return texts


def process_dictionary(
    dict_file: str,
    output_file: str,
    format_type: str = "definition",
    append: bool = False
):
    """
    辞書ファイルを処理して学習データを出力

    Args:
        dict_file: 専門用語辞書JSONファイル
        output_file: 出力JSONLファイル
        format_type: 出力形式
        append: Trueの場合、既存ファイルに追記
    """
    dict_path = Path(dict_file)
    output_path = Path(output_file)

    if not dict_path.exists():
        raise FileNotFoundError(f"辞書ファイルが見つかりません: {dict_file}")

    print(f"辞書読み込み: {dict_file}")
    entries = extract_dictionary_entries(dict_file)
    print(f"抽出用語数: {len(entries)}")

    texts = generate_training_texts(entries, format_type)
    print(f"生成テキスト数: {len(texts)}")

    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    action = "追記" if append else "出力"
    print(f"{action}完了: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="専門用語辞書から学習データを生成"
    )
    parser.add_argument(
        "--dict", "-d",
        default="data/dict/terms.json",
        help="専門用語辞書JSONファイル"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/output/dictionary.jsonl",
        help="出力JSONLファイル"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["all", "definition", "simple", "qa"],
        default="all",
        help="出力形式: all(全形式を含む), definition(用語とは...), simple(用語: ...), qa(Q&A形式)"
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="既存ファイルに追記"
    )

    args = parser.parse_args()

    process_dictionary(
        dict_file=args.dict,
        output_file=args.output,
        format_type=args.format,
        append=args.append
    )


if __name__ == "__main__":
    main()
