"""
LLMプロンプト定義モジュール

全てのLLM用プロンプトを集約管理
修正時はこのファイルのみ編集すればOK
"""

# =============================================================================
# システムプロンプト
# =============================================================================

SYSTEM_PROMPTS = {
    # 難問Q&A生成 (expand_qa_difficult.py)
    "qa_difficult": """あなたは専門的なQ&Aを生成する専門家です。

与えられたテキストと専門用語辞書の情報を基に、専門知識が必要な難しい質問と回答のペアを生成してください。

ルール:
- テキストの内容と辞書の専門用語情報を組み合わせて質問を作成する
- 質問は専門用語や業界知識を理解していないと答えられないレベルにする
- 回答は辞書の定義とテキストの内容を基に、詳細かつ正確に記述する
- 3〜5個のQ&Aペアを生成する

質問の種類の例:
- 専門用語の意味を問う質問（「〜とは何ですか」）
- 概念間の関係を問う質問（「〜と〜の関係は」）
- 技術的な仕組みを問う質問（「〜はどのように機能しますか」）
- 応用や実例を問う質問（「〜はどのような場面で使われますか」）
""",

    # 英語翻訳 (expand_to_english.py)
    "english_translation": """You are a professional translator.
Translate the given Japanese text to English accurately while preserving:
- Technical terminology
- Document structure (markdown, HTML tags, etc.)
- Numbers and data
- Formatting and line breaks

Provide natural, fluent English translation.""",

    # グラフ関係性テキスト化 (expand_graph_relations.py)
    "graph_relations": """あなたは専門用語の関係性を自然な日本語文にまとめる専門家です。

与えられた用語とその関係情報を、自然で読みやすい日本語の文章にまとめてください。

ルール:
- 関係タイプの意味を理解し、適切な日本語表現に変換する
- 同じタイプの関係は1文にまとめる（例: 「〜は〜や〜に適用される」）
- 指定された文字数程度で簡潔にまとめる
- 専門用語はそのまま使用する
- 冗長な表現を避け、情報密度の高い文章にする

関係タイプの例:
- RELATED_TO: 関連している
- APPLIES_TO: 適用される、使用される
- IS_A: 〜の一種である、〜に分類される
- AFFECTS: 影響を与える
- HAS_ATTRIBUTE: 〜という特性を持つ
- PART_OF: 〜の一部である
- BELONGS_TO_CATEGORY: 〜に分類される
""",

    # Llama-3フォーマット用 (preprocess_dataset.py)
    "llama3_assistant": "あなたは誠実で有能な日本語のAIアシスタントです。ユーザーの質問に対して、正確で役立つ回答を提供してください。",
}


# =============================================================================
# ユーザープロンプトテンプレート
# =============================================================================

USER_PROMPTS = {
    # 画像説明 (datasetmaker.py)
    "image_description": """この画像を分析してください。
もし画像が主に枠で囲まれたテキストである場合、"TEXT_ONLY:"という接頭辞に続けて、画像内のテキストを全て書き起こしてください。
それ以外の場合は、この画像の内容を統合報告書に掲載するキャプションとして、以下の形式で簡潔に日本語で説明してください。

- **種類**: 画像の種類を特定します（例: 製品写真, ウェブサイトのスクリーンショット, ロゴ, グラフ, 表）。
- **内容**: 画像に写っている主要な要素や情報を客観的に記述します。
- **示唆**: (任意) この画像がビジネスや経営において持つ可能性のある意味や示唆を簡潔に述べます。

説明は客観的かつ簡潔にまとめてください。""",

    # 言い換え (datasetmaker.py)
    "paraphrase": "以下の文章を、元の意味を完全に保持したまま、異なる表現で言い換えてください。言い換えた後の文章のみを出力してください。\n\n# 元の文章:\n{text}",

    # QA生成 (datasetmaker.py)
    "qa": """以下の文章に基づいて、質の高い質問と回答を1ペア生成してください。
出力は以下のJSON形式で:
```json
{{"question": "質問", "answer": "回答"}}
```

# 文章:
{context}""",

    # 要約 (datasetmaker.py)
    "summary": "以下の文章を{max_length}文字程度で簡潔に要約してください。\n\n# 文章:\n{text}",

    # キーワード抽出 (datasetmaker.py)
    "keywords": """以下の文章から重要なキーワードを5つ抽出し、説明を付けてください。
出力は以下のJSON形式で:
```json
[{{"keyword": "キーワード", "description": "説明"}}]
```

# 文章:
{text}""",

    # 詳細化 (datasetmaker.py)
    "elaboration": "以下の文章を、具体例や背景情報を補足しながらより詳細に説明してください。\n\n# 元の文章:\n{text}",

    # 翻訳 (datasetmaker.py)
    "translation": "以下の日本語の文章を自然な{lang}に翻訳してください。翻訳後の文章のみを出力。\n\n# 元の文章:\n{text}",

    # 特定観点QA (datasetmaker.py)
    "specialized_qa": """以下の文章から「{type_desc}」に焦点を当てたQAを1ペア生成。
出力はJSON形式: {{"question": "質問", "answer": "回答"}}

# 文章:
{context}""",

    # 議論生成 (datasetmaker.py)
    "discussion": """以下の文章について専門家同士の議論を{turns}往復で生成してください。
形式: 質問者: (内容) / 回答者: (内容)

# 元の文章:
{text}""",

    # 難問QA (expand_qa_difficult.py)
    "qa_difficult": """以下のテキストと専門用語情報を基に、専門的なQ&Aを生成してください。

【テキスト】
{text}

【関連する専門用語】
{terms_info}
""",

    # グラフ関係性 (expand_graph_relations.py)
    "graph_relations": """以下の用語とその関係情報を、自然な日本語の文章（{max_chars}文字程度）にまとめてください。

用語: {node}
関係:
{relations_formatted}
""",
}


# =============================================================================
# ヘルパー関数
# =============================================================================

def get_system_prompt(name: str) -> str:
    """
    システムプロンプトを取得

    Args:
        name: プロンプト名

    Returns:
        str: システムプロンプト
    """
    return SYSTEM_PROMPTS.get(name, "")


def get_user_prompt(name: str, **kwargs) -> str:
    """
    ユーザープロンプトを取得し、変数を埋め込む

    Args:
        name: プロンプト名
        **kwargs: テンプレート変数

    Returns:
        str: 変数埋め込み済みプロンプト

    Example:
        >>> get_user_prompt("summary", text="元のテキスト", max_length=150)
        "以下の文章を150文字程度で簡潔に要約してください。..."
    """
    template = USER_PROMPTS.get(name, "")
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"プロンプト '{name}' に必要な変数が不足しています: {e}")


# 特定観点QAの観点名マッピング
SPECIALIZED_QA_TYPES = {
    "methods": "手法・技術",
    "people": "人物名・組織名",
    "numbers": "数値・統計データ",
}


def get_specialized_qa_prompt(context: str, qa_type: str) -> str:
    """
    特定観点QA用のプロンプトを取得

    Args:
        context: 文章
        qa_type: 観点タイプ (methods, people, numbers)

    Returns:
        str: プロンプト
    """
    type_desc = SPECIALIZED_QA_TYPES.get(qa_type, "一般")
    return get_user_prompt("specialized_qa", context=context, type_desc=type_desc)
