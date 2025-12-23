"""
専門用語辞書を使って専門用語を一般的な用語に置き換えるデータ拡張スクリプト
辞書ベースで置換を行う（LLM不要）
"""

import json
import re
from pathlib import Path
from typing import Optional


def load_dictionary(dict_file: str) -> dict:
    """
    専門用語辞書を読み込み、検索用の辞書を構築

    Returns:
        dict: {用語: {"headword": str, "brief_definition": str, "definition": str}}
    """
    with open(dict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    term_dict = {}
    for term in data.get("terms", []):
        headword = term.get("headword", "")
        brief_def = term.get("brief_definition", "")
        definition = term.get("definition", "")

        if headword:
            term_dict[headword] = {
                "headword": headword,
                "brief_definition": brief_def,
                "definition": definition
            }
            # 同義語も登録
            for synonym in term.get("synonyms", []):
                if synonym:
                    term_dict[synonym] = {
                        "headword": headword,
                        "brief_definition": brief_def,
                        "definition": definition
                    }

    return term_dict


def generalize_text(text: str, term_dict: dict) -> dict:
    """
    テキスト内の専門用語を一般的な表現（brief_definition）に置き換える
    """
    result_text = text
    replacements = []

    # 長い用語から先に置換（部分一致を防ぐ）
    sorted_terms = sorted(term_dict.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term in result_text:
            info = term_dict[term]
            brief_def = info["brief_definition"]

            if brief_def:
                # 簡潔な置換表現を作成（brief_definitionの最初の文を使用）
                replacement = brief_def.split("。")[0]
                if len(replacement) > 50:
                    replacement = replacement[:50] + "..."

                # 置換を実行
                result_text = result_text.replace(term, replacement, 1)
                replacements.append({
                    "original": term,
                    "replaced": replacement,
                    "headword": info["headword"]
                })

    return {
        "text": result_text,
        "replacements": replacements
    }


def process_jsonl_file(
    input_file: str,
    output_file: str,
    dict_file: str,
    max_lines: Optional[int] = None
):
    """
    JSONLファイルを読み込み、専門用語を一般化して出力

    Args:
        input_file: 入力JSONLファイルのパス
        output_file: 出力JSONLファイルのパス
        dict_file: 専門用語辞書JSONファイルのパス
        max_lines: 処理する最大行数（テスト用、Noneで全行処理）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    # 辞書を読み込み
    print(f"辞書読み込み: {dict_file}")
    term_dict = load_dictionary(dict_file)
    print(f"登録用語数: {len(term_dict)}")

    print(f"処理開始: {input_file}")
    print(f"出力先: {output_file}")

    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            if max_lines and line_num > max_lines:
                break

            try:
                data = json.loads(line.strip())

                if "text" not in data:
                    print(f"警告: 行{line_num}に'text'フィールドがありません")
                    continue

                # 一般化処理
                result = generalize_text(data["text"], term_dict)

                output_data = {
                    "text": result["text"],
                    "augmentation_type": "generalized",
                    "replacements": result["replacements"],
                    "source_id": data.get("id", "")
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"進捗: {processed_count}件処理完了")

            except json.JSONDecodeError as e:
                print(f"JSON解析エラー (行{line_num}): {e}")
                continue

    print(f"処理完了: 合計{processed_count}件")


def main():
    """メイン処理"""
    input_file = "data/output/1_preprocessed.jsonl"
    output_file = "data/output/1_generalized.jsonl"
    dict_file = "data/dict/terms.json"  # 専門用語辞書
    max_lines = 3  # テスト時は小さい値、本番はNone

    process_jsonl_file(
        input_file=input_file,
        output_file=output_file,
        dict_file=dict_file,
        max_lines=max_lines
    )


if __name__ == "__main__":
    main()
