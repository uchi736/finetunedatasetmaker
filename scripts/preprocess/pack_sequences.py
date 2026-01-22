"""
継続事前学習用にテキストをパッキング（結合）するスクリプト
max_seq_len に収まるようにチャンクを結合して学習効率を向上

使用例:
    python scripts/preprocess/pack_sequences.py data/output/sample.jsonl -o data/output/packed.jsonl --max-seq-len 2048
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def estimate_tokens(text: str) -> int:
    """
    テキストのトークン数を概算
    日本語: 1文字 ≒ 1-2トークン（保守的に1.5で計算）
    英語/記号: 1単語 ≒ 1トークン
    """
    # シンプルに文字数ベースで概算
    # 日本語が多い場合は文字数×1.5程度がトークン数の目安
    return int(len(text) * 1.5)


def pack_texts(
    texts: list[str],
    max_seq_len: int,
    separator: str = "\n\n"
) -> list[str]:
    """
    テキストをmax_seq_len以下にパッキング

    Args:
        texts: テキストのリスト
        max_seq_len: 最大シーケンス長（トークン数）
        separator: テキスト間の区切り文字

    Returns:
        パッキングされたテキストのリスト
    """
    packed = []
    current_pack = []
    current_tokens = 0
    sep_tokens = estimate_tokens(separator)

    for text in texts:
        text_tokens = estimate_tokens(text)

        # 単体でmax_seq_lenを超える場合はそのまま出力
        if text_tokens > max_seq_len:
            # 現在のパックがあれば先に出力
            if current_pack:
                packed.append(separator.join(current_pack))
                current_pack = []
                current_tokens = 0
            # 長いテキストはそのまま出力
            packed.append(text)
            continue

        # 現在のパックに追加可能か判定
        new_tokens = current_tokens + text_tokens
        if current_pack:
            new_tokens += sep_tokens

        if new_tokens <= max_seq_len:
            # 追加可能
            current_pack.append(text)
            current_tokens = new_tokens
        else:
            # 追加不可 → 現在のパックを出力して新規開始
            if current_pack:
                packed.append(separator.join(current_pack))
            current_pack = [text]
            current_tokens = text_tokens

    # 残りを出力
    if current_pack:
        packed.append(separator.join(current_pack))

    return packed


def process_jsonl(
    input_file: str,
    output_file: str,
    max_seq_len: int,
    separator: str = "\n\n",
    shuffle: bool = False
):
    """
    JSONLファイルを読み込み、パッキングして出力
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    # テキストを読み込み
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "text" in data and data["text"]:
                    texts.append(data["text"])
            except json.JSONDecodeError:
                continue

    print(f"入力: {len(texts)}件")

    # シャッフル（オプション）
    if shuffle:
        import random
        random.shuffle(texts)

    # パッキング
    packed_texts = pack_texts(texts, max_seq_len, separator)

    # 出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in packed_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    # 統計
    total_chars_before = sum(len(t) for t in texts)
    total_chars_after = sum(len(t) for t in packed_texts)
    avg_tokens_per_pack = sum(estimate_tokens(t) for t in packed_texts) / len(packed_texts) if packed_texts else 0

    print(f"出力: {len(packed_texts)}件")
    print(f"圧縮率: {len(texts)} → {len(packed_texts)} ({len(packed_texts)/len(texts)*100:.1f}%)")
    print(f"平均トークン数/パック: {avg_tokens_per_pack:.0f} / {max_seq_len}")
    print(f"出力先: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="継続事前学習用にテキストをパッキング",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", help="入力JSONLファイル")
    parser.add_argument("-o", "--output", required=True, help="出力JSONLファイル")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="最大シーケンス長（トークン数）")
    parser.add_argument("--separator", default="\n\n",
                        help="テキスト間の区切り文字")
    parser.add_argument("--shuffle", action="store_true",
                        help="パッキング前にシャッフル")

    args = parser.parse_args()

    process_jsonl(
        input_file=args.input,
        output_file=args.output,
        max_seq_len=args.max_seq_len,
        separator=args.separator,
        shuffle=args.shuffle
    )


if __name__ == "__main__":
    main()
