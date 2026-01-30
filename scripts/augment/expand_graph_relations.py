"""
ナレッジグラフの関係性情報をテキスト化するデータ拡張スクリプト
graph.jsonからノードごとの関係をLLMで自然な日本語文に変換
"""

import json
import asyncio
import os
import re
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
class RelationText(BaseModel):
    """関係性テキストのスキーマ"""
    text: str = Field(description="関係性を説明する自然な日本語文")


# Azure OpenAI環境変数を設定
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

agent = Agent(
    "azure:gpt-4.1-mini",
    output_type=RelationText,
    system_prompt=get_system_prompt("graph_relations"),
)


def is_valid_node(node_id: str) -> bool:
    """
    有効なノードIDかどうかを判定
    64文字の16進数ハッシュはドキュメント参照なので除外
    """
    # 64文字の16進数パターンにマッチするか
    if re.match(r'^[0-9a-f]{64}$', node_id):
        return False
    return True


def load_graph(graph_file: str) -> dict:
    """
    graph.jsonを読み込み

    Returns:
        dict: グラフデータ（nodes, edges）
    """
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("graph", {})


def build_node_relations(graph: dict) -> dict:
    """
    ノードごとに関係をグループ化

    Returns:
        dict: {
            "ガス軸受": {
                "RELATED_TO": ["軸支持", "油潤滑軸受"],
                "APPLIES_TO": ["電動ターボ機械", "輸送機器"],
                ...
            },
            ...
        }
    """
    node_relations = {}
    edges = graph.get("edges", [])

    # 重複排除用のセット
    seen_edges = set()

    for edge in edges:
        rel_type = edge.get("type", "")
        source = edge.get("source", "")
        target = edge.get("target", "")

        # MENTIONSは除外（ドキュメント参照）
        if rel_type == "MENTIONS":
            continue

        # ハッシュIDのノードは除外
        if not is_valid_node(source) or not is_valid_node(target):
            continue

        # 重複チェック（同じsource-type-targetは1回のみ）
        edge_key = (source, rel_type, target)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # ノードの関係を格納
        if source not in node_relations:
            node_relations[source] = {}

        if rel_type not in node_relations[source]:
            node_relations[source][rel_type] = []

        node_relations[source][rel_type].append(target)

    return node_relations


def format_relations_for_prompt(node: str, relations: dict) -> str:
    """
    ノードの関係情報をプロンプト用にフォーマット
    """
    lines = []
    for rel_type, targets in relations.items():
        targets_str = ", ".join(targets)
        lines.append(f"- {rel_type}: {targets_str}")

    return "\n".join(lines)


async def generate_relation_text(node: str, relations: dict, max_chars: int = 100) -> str:
    """
    ノードの関係情報をLLMでテキスト化
    """
    relations_formatted = format_relations_for_prompt(node, relations)
    prompt = get_user_prompt(
        "graph_relations",
        node=node,
        relations_formatted=relations_formatted,
        max_chars=max_chars
    )

    try:
        result = await agent.run(prompt)
        return result.output.text
    except Exception as e:
        print(f"テキスト生成エラー ({node}): {e}")
        return ""


async def process(graph_file: str, max_chars: int = 100, batch_size: int = 5) -> list:
    """
    パイプライン用: グラフファイルから関係テキストを生成して返す

    Args:
        graph_file: グラフJSONファイルのパス
        max_chars: 生成テキストの目標文字数
        batch_size: 並行処理のバッチサイズ

    Returns:
        list: [{"text": str, "source": "graph_relations", "node": str}, ...]
    """
    graph = load_graph(graph_file)
    node_relations = build_node_relations(graph)

    results = []
    nodes = list(node_relations.keys())

    # バッチ処理
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]

        tasks = [
            generate_relation_text(node, node_relations[node], max_chars)
            for node in batch_nodes
        ]
        batch_results = await asyncio.gather(*tasks)

        for node, text in zip(batch_nodes, batch_results):
            if text:  # 空でない場合のみ追加
                relation_count = sum(len(targets) for targets in node_relations[node].values())
                results.append({
                    "text": text,
                    "source": "graph_relations",
                    "node": node,
                    "relation_count": relation_count
                })

        print(f"進捗: {min(i + batch_size, len(nodes))}/{len(nodes)}件処理完了")

    return results


async def process_graph_file(
    graph_file: str,
    output_file: str,
    max_chars: int = 100,
    batch_size: int = 5,
    max_nodes: Optional[int] = None
):
    """
    グラフファイルを読み込み、関係テキストを生成して出力

    Args:
        graph_file: 入力グラフJSONファイルのパス
        output_file: 出力JSONLファイルのパス
        max_chars: 生成テキストの目標文字数
        batch_size: 並行処理のバッチサイズ
        max_nodes: 処理する最大ノード数（テスト用、Noneで全ノード処理）
    """
    graph_path = Path(graph_file)
    output_path = Path(output_file)

    if not graph_path.exists():
        raise FileNotFoundError(f"グラフファイルが見つかりません: {graph_file}")

    # グラフを読み込み
    print(f"グラフ読み込み: {graph_file}")
    graph = load_graph(graph_file)
    node_relations = build_node_relations(graph)

    total_nodes = len(node_relations)
    total_edges = sum(len(targets) for rels in node_relations.values() for targets in rels.values())
    print(f"有効ノード数: {total_nodes}")
    print(f"有効エッジ数: {total_edges}")

    print(f"処理開始")
    print(f"出力先: {output_file}")
    print(f"目標文字数: {max_chars}文字")

    processed_count = 0
    nodes = list(node_relations.keys())

    if max_nodes:
        nodes = nodes[:max_nodes]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        # バッチ処理
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]

            tasks = [
                generate_relation_text(node, node_relations[node], max_chars)
                for node in batch_nodes
            ]
            batch_results = await asyncio.gather(*tasks)

            for node, text in zip(batch_nodes, batch_results):
                if text:  # 空でない場合のみ出力
                    relation_count = sum(len(targets) for targets in node_relations[node].values())
                    output_data = {
                        "text": text,
                        "augmentation_type": "graph_relations",
                        "node": node,
                        "relation_count": relation_count,
                        "relations": node_relations[node]
                    }
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    processed_count += 1

            print(f"進捗: {min(i + batch_size, len(nodes))}/{len(nodes)}件処理完了")

    print(f"処理完了: 合計{processed_count}件のテキストを生成")


async def main():
    """メイン処理"""
    graph_file = "data/graph/graph.json"
    output_file = "data/output/graph_relations.jsonl"
    max_chars = 100  # 生成テキストの目標文字数
    batch_size = 5
    max_nodes = 5  # テスト時は小さい値、本番はNone

    await process_graph_file(
        graph_file=graph_file,
        output_file=output_file,
        max_chars=max_chars,
        batch_size=batch_size,
        max_nodes=max_nodes
    )


if __name__ == "__main__":
    asyncio.run(main())
