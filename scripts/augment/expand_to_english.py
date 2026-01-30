"""
JSONL形式の継続事前学習用データセットを英語に拡張するスクリプト
PydanticAIを使用してJSON形式を保証
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from openai import AzureOpenAI
from dotenv import load_dotenv

# プロンプト定義のインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import get_system_prompt

# .envファイルを読み込み（スクリプト位置基準の絶対パス）
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / 'config' / '.env')

# 出力スキーマの定義
class TranslatedText(BaseModel):
    """翻訳されたテキストのスキーマ"""
    text: str = Field(description="英語に翻訳されたテキスト")


# Azure OpenAI環境変数を設定
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

agent = Agent(
    "azure:gpt-4.1-mini",
    output_type=TranslatedText,
    system_prompt=get_system_prompt("english_translation"),
)


async def translate_text(text: str) -> str:
    """テキストを英語に翻訳"""
    try:
        result = await agent.run(text)
        return result.output.text
    except Exception as e:
        print(f"翻訳エラー: {e}")
        return text  # エラー時は元のテキストを返す


async def process(data: list, batch_size: int = 10) -> list:
    """
    パイプライン用: データリストを英語に翻訳して返す

    Args:
        data: [{"text": str, ...}, ...]
        batch_size: 並行処理のバッチサイズ

    Returns:
        list: [{"text": str, "source": "english"}, ...]
    """
    results = []
    batch = []

    for item in data:
        if "text" not in item:
            continue

        batch.append(item["text"])

        # バッチサイズに達したら処理
        if len(batch) >= batch_size:
            tasks = [translate_text(text) for text in batch]
            translated_texts = await asyncio.gather(*tasks)

            for translated in translated_texts:
                results.append({
                    "text": translated,
                    "source": "english"
                })

            batch = []

    # 残りのバッチを処理
    if batch:
        tasks = [translate_text(text) for text in batch]
        translated_texts = await asyncio.gather(*tasks)

        for translated in translated_texts:
            results.append({
                "text": translated,
                "source": "english"
            })

    return results


async def process_jsonl_file(
    input_file: str,
    output_file: str,
    batch_size: int = 10,
    max_lines: Optional[int] = None
):
    """
    JSONLファイルを読み込み、英語に拡張して出力

    Args:
        input_file: 入力JSONLファイルのパス
        output_file: 出力JSONLファイルのパス
        batch_size: 並行処理のバッチサイズ
        max_lines: 処理する最大行数（テスト用、Noneで全行処理）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    print(f"処理開始: {input_file}")
    print(f"出力先: {output_file}")

    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        batch = []
        batch_original = []

        for line_num, line in enumerate(f_in, 1):
            if max_lines and line_num > max_lines:
                break

            try:
                data = json.loads(line.strip())

                # "text"フィールドを抽出
                if "text" not in data:
                    print(f"警告: 行{line_num}に'text'フィールドがありません")
                    continue

                batch.append(data["text"])
                batch_original.append(data)

                # バッチサイズに達したら処理
                if len(batch) >= batch_size:
                    await process_batch(batch, batch_original, f_out)
                    processed_count += len(batch)
                    print(f"進捗: {processed_count}件処理完了")

                    batch = []
                    batch_original = []

            except json.JSONDecodeError as e:
                print(f"JSON解析エラー (行{line_num}): {e}")
                continue

        # 残りのバッチを処理
        if batch:
            await process_batch(batch, batch_original, f_out)
            processed_count += len(batch)

    print(f"処理完了: 合計{processed_count}件")


async def process_batch(texts: list[str], original_data: list[dict], f_out):
    """バッチ処理で翻訳を実行"""
    # 並行して翻訳を実行
    tasks = [translate_text(text) for text in texts]
    translated_texts = await asyncio.gather(*tasks)

    # 結果をJSONL形式で出力
    for translated in translated_texts:
        # text:のみのシンプルな形式
        output_data = {"text": translated}

        # JSONLとして書き込み
        f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')


async def main():
    """メイン処理"""
    # 設定
    input_file = "data/output/1_preprocessed.jsonl"  # 入力ファイルパス
    output_file = "data/output/1_preprocessed_en.jsonl"  # 出力ファイルパス
    batch_size = 5  # 並行処理数
    max_lines = 3  # テスト時は10など指定、本番はNone

    await process_jsonl_file(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        max_lines=max_lines
    )


if __name__ == "__main__":
    asyncio.run(main())
