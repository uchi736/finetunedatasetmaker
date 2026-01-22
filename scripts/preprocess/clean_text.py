"""
PDF抽出テキストのクリーニングモジュール

機能:
- Unicode正規化（NFKC）
- ページ番号・章番号除去
- ヘッダ/フッタ除去
- 目次検出・除去
- 改行修復（単語分断の修正）
- 数値・単位正規化
- 断片文フィルタ
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


@dataclass
class CleanResult:
    """クリーニング結果"""
    text: str
    removed_lines: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class TextCleaner:
    """
    PDF抽出テキストのクリーニングクラス

    使用例:
        cleaner = TextCleaner(level="aggressive")
        result = cleaner.clean(text)
        print(result.text)
    """

    def __init__(
        self,
        level: str = "basic",  # off, basic, aggressive
        remove_toc: bool = True,
        remove_page_numbers: bool = True,
        normalize_unicode: bool = True,
        normalize_numbers: bool = True,
        fix_line_breaks: bool = True,
    ):
        self.level = level
        self.remove_toc = remove_toc
        self.remove_page_numbers_flag = remove_page_numbers
        self.normalize_unicode_flag = normalize_unicode
        self.normalize_numbers_flag = normalize_numbers
        self.fix_line_breaks_flag = fix_line_breaks

        # ページ番号パターン
        self.page_number_patterns = [
            r'^[\s]*\d{1,4}[\s]*$',                      # 単独の数字行（1-9999）
            r'^[\s]*[-–—]\s*\d{1,4}\s*[-–—][\s]*$',      # - 14 -
            r'^[\s]*\d{1,4}\s*/\s*\d{1,4}[\s]*$',        # 14/100
            r'^[\s]*Page\s*\d+[\s]*$',                   # Page 14
            r'^[\s]*p\.\s*\d+[\s]*$',                    # p. 14
            r'^\s*\d+\s*ページ\s*$',                     # 14ページ
        ]

        # ヘッダ/フッタのよくあるパターン
        self.header_footer_patterns = [
            r'.*技報\s*Vol\.\s*\d+.*',                   # IHI技報Vol.62...
            r'.*\d{4}年\d{1,2}月.*発行.*',               # 2023年10月発行
            r'^第\d+[章節条項]\s*$',                     # 第1章（単独行）
            r'^\s*Copyright\s*©.*$',                    # Copyright
            r'^\s*All\s+[Rr]ights\s+[Rr]eserved.*$',    # All rights reserved
        ]

        # 目次パターン
        self.toc_patterns = [
            r'^[\s]*目[\s]*次[\s]*$',                    # 目次
            r'^[\s]*CONTENTS[\s]*$',                     # CONTENTS
            r'^[\s]*Table\s+of\s+Contents[\s]*$',        # Table of Contents
        ]

        # 目次内の行パターン（ドットリーダー+ページ番号）
        self.toc_line_patterns = [
            r'\.{3,}\s*\d+\s*$',                         # ...14
            r'…+\s*\d+\s*$',                             # …14
            r'\s+\d+\s*$',                               # 末尾がページ番号
        ]

    def clean(self, text: str, pages: Optional[list[str]] = None) -> CleanResult:
        """
        テキストをクリーニング

        Args:
            text: クリーニング対象のテキスト
            pages: ページごとに分割されたテキスト（ヘッダ/フッタ除去用）

        Returns:
            CleanResult: クリーニング結果
        """
        if self.level == "off":
            return CleanResult(text=text)

        removed_lines = []
        stats = {"original_chars": len(text)}

        # 1. Unicode正規化
        if self.normalize_unicode_flag:
            text = self._normalize_unicode(text)

        # 2. ページ番号除去
        if self.remove_page_numbers_flag:
            text, removed = self._remove_page_numbers(text)
            removed_lines.extend(removed)

        # 3. ヘッダ/フッタ除去
        if pages and self.level == "aggressive":
            text = self._remove_headers_footers_from_pages(pages)
        else:
            text, removed = self._remove_common_headers_footers(text)
            removed_lines.extend(removed)

        # 4. 目次除去
        if self.remove_toc:
            text, removed = self._remove_toc(text)
            removed_lines.extend(removed)

        # 5. 改行修復
        if self.fix_line_breaks_flag:
            text = self._fix_line_breaks(text)

        # 6. 数値・単位正規化
        if self.normalize_numbers_flag:
            text = self._normalize_numbers(text)

        # 7. 空行の正規化
        text = self._normalize_whitespace(text)

        stats["cleaned_chars"] = len(text)
        stats["removed_lines"] = len(removed_lines)
        stats["reduction_rate"] = 1 - (len(text) / max(stats["original_chars"], 1))

        return CleanResult(text=text, removed_lines=removed_lines, stats=stats)

    def _normalize_unicode(self, text: str) -> str:
        """Unicode正規化（NFKC）"""
        # NFKC: 互換分解 → 合成正規化
        # - 全角英数字 → 半角
        # - ㈱ → (株)
        # - ① → 1
        # - ｶﾀｶﾅ → カタカナ
        return unicodedata.normalize('NFKC', text)

    def _remove_page_numbers(self, text: str) -> tuple[str, list[str]]:
        """ページ番号行を除去"""
        lines = text.split('\n')
        cleaned_lines = []
        removed = []

        for line in lines:
            is_page_number = False
            for pattern in self.page_number_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_page_number = True
                    removed.append(line)
                    break

            if not is_page_number:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), removed

    def _remove_common_headers_footers(self, text: str) -> tuple[str, list[str]]:
        """よくあるヘッダ/フッタパターンを除去"""
        lines = text.split('\n')
        cleaned_lines = []
        removed = []

        for line in lines:
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_header_footer = True
                    removed.append(line)
                    break

            if not is_header_footer:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), removed

    def _remove_headers_footers_from_pages(self, pages: list[str]) -> str:
        """
        ページ単位でヘッダ/フッタを検出・除去
        50%以上のページで出現する先頭/末尾行を除去
        """
        if len(pages) < 3:
            return '\n\n'.join(pages)

        # 各ページの先頭行と末尾行を収集
        first_lines = []
        last_lines = []

        for page in pages:
            lines = [l.strip() for l in page.split('\n') if l.strip()]
            if lines:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        # 出現頻度をカウント
        first_counter = Counter(first_lines)
        last_counter = Counter(last_lines)

        threshold = len(pages) * 0.5

        # 除去対象のヘッダ/フッタ
        headers_to_remove = {line for line, count in first_counter.items() if count >= threshold}
        footers_to_remove = {line for line, count in last_counter.items() if count >= threshold}

        # 各ページからヘッダ/フッタを除去
        cleaned_pages = []
        for page in pages:
            lines = page.split('\n')
            cleaned_lines = []

            for i, line in enumerate(lines):
                stripped = line.strip()

                # 先頭行でヘッダの場合はスキップ
                if i == 0 and stripped in headers_to_remove:
                    continue

                # 末尾行でフッタの場合はスキップ
                if i == len(lines) - 1 and stripped in footers_to_remove:
                    continue

                cleaned_lines.append(line)

            cleaned_pages.append('\n'.join(cleaned_lines))

        return '\n\n'.join(cleaned_pages)

    def _remove_toc(self, text: str) -> tuple[str, list[str]]:
        """目次セクションを検出・除去"""
        lines = text.split('\n')
        cleaned_lines = []
        removed = []
        in_toc = False
        toc_end_blank_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 目次の開始を検出
            if not in_toc:
                for pattern in self.toc_patterns:
                    if re.match(pattern, stripped, re.IGNORECASE):
                        in_toc = True
                        removed.append(line)
                        break

                if not in_toc:
                    cleaned_lines.append(line)
                continue

            # 目次内の処理
            if in_toc:
                # 空行カウント
                if not stripped:
                    toc_end_blank_count += 1
                    if toc_end_blank_count >= 2:
                        # 連続した空行で目次終了
                        in_toc = False
                        toc_end_blank_count = 0
                        cleaned_lines.append(line)
                    else:
                        removed.append(line)
                    continue

                toc_end_blank_count = 0

                # 目次行パターンにマッチするか
                is_toc_line = False
                for pattern in self.toc_line_patterns:
                    if re.search(pattern, stripped):
                        is_toc_line = True
                        break

                # 短い行（章タイトルのみ）も目次の可能性
                if not is_toc_line and len(stripped) < 50:
                    # 次の行がドットリーダーを含むかチェック
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        for pattern in self.toc_line_patterns:
                            if re.search(pattern, next_line):
                                is_toc_line = True
                                break

                if is_toc_line:
                    removed.append(line)
                else:
                    # 目次パターンでなければ目次終了
                    in_toc = False
                    cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), removed

    def _fix_line_breaks(self, text: str) -> str:
        """
        PDF行折り返しによる単語分断を修復
        - 日本語文字間の不要な改行を除去
        - 句点で終わらない行の連結
        """
        # 日本語文字間の改行を除去（既存の clean_japanese_text と同様）
        text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])\n+([ぁ-んァ-ヴー一-龠々〆〤])', r'\1\2', text)

        # 日本語文字の後の改行+スペース+日本語文字
        text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])\n+\s*([ぁ-んァ-ヴー一-龠々〆〤])', r'\1\2', text)

        # 文中の不要な改行を検出（句読点で終わらない行）
        lines = text.split('\n')
        result_lines = []
        buffer = ""

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if buffer:
                    result_lines.append(buffer)
                    buffer = ""
                result_lines.append(line)
                continue

            # 見出し行（#で始まる）は単独で保持
            if stripped.startswith('#'):
                if buffer:
                    result_lines.append(buffer)
                    buffer = ""
                result_lines.append(line)
                continue

            # バッファに追加
            if buffer:
                # 前の行が句読点で終わっていない場合は連結
                if not buffer.rstrip().endswith(('。', '．', '？', '！', ')', '）', '」', '』')):
                    buffer = buffer.rstrip() + stripped
                else:
                    result_lines.append(buffer)
                    buffer = stripped
            else:
                buffer = stripped

        if buffer:
            result_lines.append(buffer)

        return '\n'.join(result_lines)

    def _normalize_numbers(self, text: str) -> str:
        """数値・単位の正規化"""
        # 数値内のスペースを除去: "1/1 000" → "1/1000"
        text = re.sub(r'(\d)\s+(\d{3})', r'\1\2', text)

        # 分数のスペース正規化: "1 / 1000" → "1/1000"
        text = re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2', text)

        # 単位前のスペース統一（オプション: スペースを残す場合はコメントアウト）
        # text = re.sub(r'(\d)\s+(kPa|MPa|Pa|kg|g|mg|m|cm|mm|μm|nm|L|mL|°C|K|Hz|kHz|MHz|GHz)', r'\1\2', text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """空白・空行の正規化"""
        # 3つ以上の連続した改行を2つに
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 行末のスペースを除去
        text = re.sub(r' +\n', '\n', text)

        # 連続するスペースを1つに
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def is_complete_sentence(self, text: str, min_chars: int = 20) -> bool:
        """
        完全な文かどうかを判定（断片文フィルタ用）

        Args:
            text: 判定対象のテキスト
            min_chars: 最小文字数

        Returns:
            bool: 完全な文ならTrue
        """
        if len(text) < min_chars:
            return False

        # 句点率チェック（日本語の場合、1文あたり平均50-100文字程度）
        punctuation_count = text.count('。') + text.count('．') + text.count('？') + text.count('！')
        if len(text) > 100 and punctuation_count == 0:
            return False

        # 括弧の対応チェック
        open_parens = text.count('(') + text.count('（') + text.count('「') + text.count('『')
        close_parens = text.count(')') + text.count('）') + text.count('」') + text.count('』')
        if abs(open_parens - close_parens) > 2:
            return False

        # 助詞・接続詞で終わる（不完全な文）
        incomplete_endings = ['が', 'を', 'に', 'で', 'と', 'は', 'も', 'へ', 'や', 'の', 'から', 'まで', 'より', 'ので', 'ため', 'けど', 'けれど', 'ところ']
        stripped = text.rstrip()
        for ending in incomplete_endings:
            if stripped.endswith(ending):
                return False

        return True

    def filter_fragments(self, chunks: list[dict], min_chars: int = 50) -> list[dict]:
        """
        断片的なチャンクをフィルタリング

        Args:
            chunks: チャンクのリスト（{"text": str, ...}形式）
            min_chars: 最小文字数

        Returns:
            list: フィルタリング後のチャンクリスト
        """
        filtered = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if self.is_complete_sentence(text, min_chars):
                filtered.append(chunk)

        return filtered


def process(input_file: str, output_file: str, level: str = "basic", **kwargs):
    """
    JSONLファイルを読み込み、クリーニングして出力

    Args:
        input_file: 入力JSONLファイル
        output_file: 出力JSONLファイル
        level: クリーニングレベル（off, basic, aggressive）
    """
    import json
    from pathlib import Path

    cleaner = TextCleaner(level=level, **kwargs)

    input_path = Path(input_file)
    output_path = Path(output_file)

    processed_count = 0
    total_stats = {"original_chars": 0, "cleaned_chars": 0}

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())

                if "text" not in data:
                    continue

                result = cleaner.clean(data["text"])
                data["text"] = result.text

                # 断片文フィルタ
                if not cleaner.is_complete_sentence(result.text):
                    continue

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                total_stats["original_chars"] += result.stats.get("original_chars", 0)
                total_stats["cleaned_chars"] += result.stats.get("cleaned_chars", 0)
                processed_count += 1

            except json.JSONDecodeError:
                continue

    reduction = 1 - (total_stats["cleaned_chars"] / max(total_stats["original_chars"], 1))
    print(f"処理完了: {processed_count}件")
    print(f"文字数削減率: {reduction:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="テキストクリーニング")
    parser.add_argument("input", help="入力JSONLファイル")
    parser.add_argument("-o", "--output", required=True, help="出力JSONLファイル")
    parser.add_argument("--level", choices=["off", "basic", "aggressive"], default="basic", help="クリーニングレベル")
    parser.add_argument("--no-toc", action="store_true", help="目次除去を無効化")
    parser.add_argument("--no-page-numbers", action="store_true", help="ページ番号除去を無効化")

    args = parser.parse_args()

    process(
        args.input,
        args.output,
        level=args.level,
        remove_toc=not args.no_toc,
        remove_page_numbers=not args.no_page_numbers,
    )
