"""
専門用語辞書を使って重要なキーワードの抽出とその説明を生成するデータ拡張スクリプト
辞書ベースでキーワード情報を抽出（LLM不要）
"""

import json
from pathlib import Path
from typing import Optional


def load_dictionary(dict_file: str) -> dict:
    """
    専門用語辞書を読み込み、検索用の辞書を構築

    Returns:
        dict: {用語: 用語情報dict}
    """
    with open(dict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    term_dict = {}
    for term in data.get("terms", []):
        headword = term.get("headword", "")
        if headword:
            term_dict[headword] = term
            # 同義語も登録（見出し語の情報を参照）
            for synonym in term.get("synonyms", []):
                if synonym:
                    term_dict[synonym] = term

    return term_dict


def extract_keywords(text: str, term_dict: dict) -> dict:
    """
    テキスト内に出現する専門用語を辞書から抽出し、詳細情報を付与する
    """
    keywords = []
    seen_headwords = set()

    # 長い用語から先に検索（部分一致を防ぐ）
    sorted_terms = sorted(term_dict.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term in text:
            info = term_dict[term]
            headword = info.get("headword", term)

            # 同じ見出し語は1回のみ
            if headword in seen_headwords:
                continue

            keywords.append({
                "term": headword,
                "synonyms": info.get("synonyms", []),
                "domain": info.get("domain", ""),
                "brief_definition": info.get("brief_definition", ""),
                "definition": info.get("definition", ""),
                "confidence": info.get("confidence", 1.0)
            })
            seen_headwords.add(headword)

    # confidence順でソート
    keywords.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    return {"keywords": keywords}


def generate_keywords_text(keywords: list[dict]) -> str:
    """キーワード情報をマークダウン形式のテキストに変換"""
    if not keywords:
        return ""

    parts = ["## 専門用語解説\n"]

    for kw in keywords:
        term = kw.get("term", "")
        synonyms = kw.get("synonyms", [])
        domain = kw.get("domain", "")
        brief_def = kw.get("brief_definition", "")
        definition = kw.get("definition", "")

        # 見出し
        synonyms_str = f"（別名: {', '.join(synonyms)}）" if synonyms else ""
        parts.append(f"### {term}{synonyms_str}\n")

        # 分野
        if domain:
            parts.append(f"- **分野**: {domain}\n")

        # 簡潔な定義
        if brief_def:
            parts.append(f"- **概要**: {brief_def}\n")

        # 詳細な定義
        if definition:
            parts.append(f"- **詳細**: {definition}\n")

        parts.append("")

    return "\n".join(parts)


def process_jsonl_file(
    input_file: str,
    output_file: str,
    dict_file: str,
    max_lines: Optional[int] = None
):
    """
    JSONLファイルを読み込み、キーワードを抽出して出力

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

                # キーワード抽出
                result = extract_keywords(data["text"], term_dict)
                keywords = result["keywords"]

                if keywords:  # キーワードが見つかった場合のみ出力
                    keywords_text = generate_keywords_text(keywords)

                    output_data = {
                        "text": keywords_text,
                        "augmentation_type": "keywords",
                        "keywords": keywords,
                        "keyword_count": len(keywords),
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
    output_file = "data/output/1_keywords.jsonl"
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
