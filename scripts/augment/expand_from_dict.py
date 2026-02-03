"""
専門用語辞書から直接学習データを生成するスクリプト
辞書のheadwordとdefinitionを使って複数パターンの学習テキストを生成
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def dict_to_training_data(dict_file: str) -> List[Dict[str, str]]:
    """
    辞書エントリから学習データを生成

    Args:
        dict_file: 辞書ファイルのパス

    Returns:
        list: [{"text": str}, ...] 形式の学習データリスト
    """
    with open(dict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for term in data.get("terms", []):
        headword = term.get("headword", "")
        definition = term.get("definition", "")
        brief = term.get("brief_definition", "")

        if not headword or not definition:
            continue

        # パターン1: 定義文
        results.append({"text": f"{headword}とは、{definition}"})

        # パターン2: Q&A形式
        results.append({"text": f"質問: {headword}とは何ですか？\n回答: {definition}"})

        # パターン3: 説明文
        results.append({"text": f"{headword}について説明します。{definition}"})

        # パターン4: 簡潔な説明（brief_definitionがある場合）
        if brief:
            results.append({"text": f"{headword}は{brief}です。"})
            # 簡潔Q&A
            results.append({"text": f"質問: {headword}を簡潔に説明してください。\n回答: {brief}"})

        # パターン5: 同義語がある場合
        synonyms = term.get("synonyms", [])
        if synonyms:
            syn_text = "、".join(synonyms)
            results.append({"text": f"{headword}は{syn_text}とも呼ばれます。{definition}"})

    return results


def process(dict_file: str) -> List[Dict[str, str]]:
    """
    パイプライン用: 辞書から学習データを生成して返す

    Args:
        dict_file: 辞書ファイルのパス

    Returns:
        list: [{"text": str, "source": "dictionary"}, ...]
    """
    results = dict_to_training_data(dict_file)
    # sourceフィールドを追加
    return [{"text": item["text"], "source": "dictionary"} for item in results]


def process_dict_file(dict_file: str, output_file: str = None) -> List[Dict[str, str]]:
    """
    辞書ファイルを処理して学習データを生成

    Args:
        dict_file: 入力辞書ファイルのパス
        output_file: 出力JSONLファイルのパス（省略時は標準出力）

    Returns:
        生成されたデータのリスト
    """
    results = dict_to_training_data(dict_file)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  辞書から{len(results)}件の学習データを生成 -> {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="専門用語辞書から学習データを生成")
    parser.add_argument("dict_file", help="入力辞書ファイル (JSON)")
    parser.add_argument("-o", "--output", help="出力ファイル (JSONL)")
    args = parser.parse_args()

    results = process_dict_file(args.dict_file, args.output)

    if not args.output:
        # 出力ファイル未指定時はサマリーを表示
        print(f"生成データ数: {len(results)}")
        print("\nサンプル (最初の5件):")
        for item in results[:5]:
            print(f"  - {item['text'][:80]}...")
