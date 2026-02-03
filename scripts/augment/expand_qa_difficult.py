"""
専門用語辞書を使って専門知識が必要な難しい質問（QA）を生成するデータ拡張スクリプト
辞書の用語情報を基にLLMでQ&Aを生成
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

# プロンプト定義のインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import get_system_prompt, get_user_prompt

# .envファイルを読み込み（スクリプト位置基準の絶対パス）
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / 'config' / '.env')


# 出力スキーマの定義
class QAPair(BaseModel):
    """Q&Aペアのスキーマ"""
    question: str = Field(description="専門知識が必要な質問")
    answer: str = Field(description="詳細な回答")

class QAOutput(BaseModel):
    """QA生成の出力スキーマ"""
    qa_pairs: list[QAPair] = Field(description="生成されたQ&Aペアのリスト")


# Azure OpenAI環境変数を設定
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

_model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
agent = Agent(
    f"azure:{_model_name}",
    output_type=QAOutput,
    system_prompt=get_system_prompt("qa_difficult"),
)


def load_dictionary(dict_file: str) -> tuple[dict, list]:
    """
    専門用語辞書を読み込み

    Returns:
        tuple: (検索用辞書, 全用語リスト)
    """
    with open(dict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    term_dict = {}
    all_terms = []

    for term in data.get("terms", []):
        headword = term.get("headword", "")
        if headword:
            term_dict[headword] = term
            all_terms.append(term)
            for synonym in term.get("synonyms", []):
                if synonym:
                    term_dict[synonym] = term

    return term_dict, all_terms


def find_terms_in_text(text: str, term_dict: dict) -> list[dict]:
    """テキスト内に出現する専門用語を検索"""
    found_terms = []
    seen_headwords = set()

    sorted_terms = sorted(term_dict.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term in text:
            info = term_dict[term]
            headword = info.get("headword", term)
            if headword not in seen_headwords:
                found_terms.append(info)
                seen_headwords.add(headword)

    return found_terms


async def generate_qa(text: str, terms: list[dict]) -> list[dict]:
    """テキストと専門用語情報からQ&Aを生成"""
    if not terms:
        return []

    # 用語情報をプロンプト用に整形
    terms_info = "\n".join([
        f"- {t.get('headword', '')}: {t.get('brief_definition', '')}"
        for t in terms[:10]  # 最大10用語
    ])

    prompt = get_user_prompt("qa_difficult", text=text[:2000], terms_info=terms_info)

    try:
        result = await agent.run(prompt)
        return [
            {
                "question": qa.question,
                "answer": qa.answer
            }
            for qa in result.output.qa_pairs
        ]
    except Exception as e:
        print(f"QA生成エラー: {e}")
        return []


async def process(data: list, dict_file: str, batch_size: int = 5) -> list:
    """
    パイプライン用: データリストからQ&Aを生成して返す

    Args:
        data: [{"text": str, ...}, ...]
        dict_file: 専門用語辞書JSONファイル
        batch_size: 並行処理のバッチサイズ

    Returns:
        list: [{"text": str, "source": "qa"}, ...]
    """
    term_dict, _ = load_dictionary(dict_file)
    results = []

    # バッチ処理用のデータを準備
    batch_texts = []
    batch_terms = []

    for item in data:
        if "text" not in item:
            continue

        text = item["text"]
        found_terms = find_terms_in_text(text, term_dict)

        if found_terms:
            batch_texts.append(text)
            batch_terms.append(found_terms)

        # バッチサイズに達したら処理
        if len(batch_texts) >= batch_size:
            tasks = [generate_qa(text, terms) for text, terms in zip(batch_texts, batch_terms)]
            batch_results = await asyncio.gather(*tasks)

            for qa_pairs in batch_results:
                for qa in qa_pairs:
                    qa_text = f"質問: {qa['question']}\n\n回答: {qa['answer']}"
                    results.append({
                        "text": qa_text,
                        "source": "qa"
                    })

            batch_texts = []
            batch_terms = []

    # 残りのバッチを処理
    if batch_texts:
        tasks = [generate_qa(text, terms) for text, terms in zip(batch_texts, batch_terms)]
        batch_results = await asyncio.gather(*tasks)

        for qa_pairs in batch_results:
            for qa in qa_pairs:
                qa_text = f"質問: {qa['question']}\n\n回答: {qa['answer']}"
                results.append({
                    "text": qa_text,
                    "source": "qa"
                })

    return results


async def process_jsonl_file(
    input_file: str,
    output_file: str,
    dict_file: str,
    batch_size: int = 5,
    max_lines: Optional[int] = None
):
    """
    JSONLファイルを読み込み、専門的なQ&Aを生成して出力
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    # 辞書を読み込み
    print(f"辞書読み込み: {dict_file}")
    term_dict, all_terms = load_dictionary(dict_file)
    print(f"登録用語数: {len(term_dict)}")

    print(f"処理開始: {input_file}")
    print(f"出力先: {output_file}")

    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        batch_texts = []
        batch_terms = []
        batch_data = []

        for line_num, line in enumerate(f_in, 1):
            if max_lines and line_num > max_lines:
                break

            try:
                data = json.loads(line.strip())

                if "text" not in data:
                    print(f"警告: 行{line_num}に'text'フィールドがありません")
                    continue

                text = data["text"]
                found_terms = find_terms_in_text(text, term_dict)

                if found_terms:  # 専門用語が見つかった場合のみ処理
                    batch_texts.append(text)
                    batch_terms.append(found_terms)
                    batch_data.append(data)

                if len(batch_texts) >= batch_size:
                    await process_batch(batch_texts, batch_terms, batch_data, f_out)
                    processed_count += len(batch_texts)
                    print(f"進捗: {processed_count}件処理完了")

                    batch_texts = []
                    batch_terms = []
                    batch_data = []

            except json.JSONDecodeError as e:
                print(f"JSON解析エラー (行{line_num}): {e}")
                continue

        # 残りのバッチを処理
        if batch_texts:
            await process_batch(batch_texts, batch_terms, batch_data, f_out)
            processed_count += len(batch_texts)

    print(f"処理完了: 合計{processed_count}件")


async def process_batch(
    texts: list[str],
    terms_list: list[list[dict]],
    original_data: list[dict],
    f_out
):
    """バッチ処理でQA生成を実行"""
    tasks = [generate_qa(text, terms) for text, terms in zip(texts, terms_list)]
    results = await asyncio.gather(*tasks)

    for i, qa_pairs in enumerate(results):
        for qa in qa_pairs:
            qa_text = f"質問: {qa['question']}\n\n回答: {qa['answer']}"
            output_data = {
                "text": qa_text,
                "augmentation_type": "qa_difficult",
                "question": qa["question"],
                "answer": qa["answer"],
                "related_terms": [t.get("headword", "") for t in terms_list[i]],
                "source_id": original_data[i].get("id", "")
            }
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')


async def main():
    """メイン処理"""
    input_file = "data/output/1_preprocessed.jsonl"
    output_file = "data/output/1_qa_difficult.jsonl"
    dict_file = "data/dict/terms.json"  # 専門用語辞書
    batch_size = 5
    max_lines = 3  # テスト時は小さい値、本番はNone

    await process_jsonl_file(
        input_file=input_file,
        output_file=output_file,
        dict_file=dict_file,
        batch_size=batch_size,
        max_lines=max_lines
    )


if __name__ == "__main__":
    asyncio.run(main())
