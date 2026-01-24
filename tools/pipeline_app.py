"""
LLM学習データセット作成パイプライン GUI

起動: streamlit run tools/pipeline_app.py
"""

import streamlit as st
import subprocess
import json
import os
import sys
from pathlib import Path

# ページ設定
st.set_page_config(
    page_title="Dataset Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .command-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.markdown("""
<div class="main-header">
    <h1>📊 LLM学習データセット作成パイプライン</h1>
    <p>PDF → JSONL 変換・拡張・パッキング</p>
</div>
""", unsafe_allow_html=True)

# タブ構成
tab1, tab2, tab3 = st.tabs(["📦 バッチパイプライン", "⚙️ 個別スクリプト", "👁️ ビューアー"])

# =============================================================================
# Tab 1: バッチパイプライン
# =============================================================================
with tab1:
    # PDFファイル数を取得
    input_dir_default = "data/input"
    try:
        pdf_count = len(list(Path(input_dir_default).glob("*.pdf")))
    except:
        pdf_count = 0

    # メトリクスカード
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📁 入力PDF", f"{pdf_count}件")
    with m2:
        st.metric("📄 出力形式", "JSONL")
    with m3:
        st.metric("🔧 モード", "バッチ処理")
    with m4:
        st.metric("📊 ステータス", "準備完了")

    st.divider()

    # 入力モード選択（フォーム外 - モード切替時のみ再描画）
    input_mode = st.radio(
        "入力モード",
        ["📁 PDFフォルダ", "📄 JSONLファイル"],
        horizontal=True,
        help="PDFから新規作成 or 既存JSONLを拡張",
        key="input_mode_radio"
    )

    # JSONLファイルリストを事前取得（フォーム外で実行）
    jsonl_search_dirs = ["data/output", "data", "."]
    jsonl_files = []
    for search_dir in jsonl_search_dirs:
        if Path(search_dir).exists():
            jsonl_files.extend(Path(search_dir).glob("*.jsonl"))
            jsonl_files.extend(Path(search_dir).glob("**/*.jsonl"))
    jsonl_files = sorted(set(str(f) for f in jsonl_files))

    # モード判定（フォーム内外で使用）
    is_pdf_mode = "PDF" in input_mode

    # 設定セクション（フォームなし - 条件分岐の問題を回避）
    # 入出力設定
    with st.expander("📂 入出力設定", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # PDFモード用入力
            input_path_pdf = st.text_input(
                "入力フォルダ (PDF)",
                value=input_dir_default,
                key="batch_input",
                help="PDFファイルが格納されているフォルダ",
                disabled=not is_pdf_mode
            )
            # JSONLモード用入力
            if jsonl_files:
                input_path_jsonl = st.selectbox(
                    "入力JSONLファイル",
                    options=[""] + jsonl_files,
                    key="batch_input_jsonl",
                    help="拡張したいJSONLファイルを選択",
                    disabled=is_pdf_mode
                )
            else:
                input_path_jsonl = st.text_input(
                    "入力JSONLファイル",
                    value="data/output/preprocessed.jsonl",
                    key="batch_input_jsonl_manual",
                    help="拡張したいJSONLファイルのパス",
                    disabled=is_pdf_mode
                )
            # 実際に使用するパスを決定
            input_path = input_path_pdf if is_pdf_mode else input_path_jsonl
        with col2:
            output_file = st.text_input(
                "出力ファイル",
                value="data/output/train.jsonl",
                key="batch_output",
                help="生成されるJSONLファイルのパス"
            )

    # PDF処理設定（常に表示、JSONLモード時は無効）
    with st.expander("🔧 PDF処理設定", expanded=False):
        if not is_pdf_mode:
            st.caption("💡 PDFモード選択時に有効になります")
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input(
                "チャンクサイズ",
                value=1500,
                min_value=100,
                max_value=10000,
                step=100,
                help="テキストを分割する際の最大文字数",
                disabled=not is_pdf_mode
            )
        with col2:
            chunk_overlap = st.number_input(
                "オーバーラップ",
                value=100,
                min_value=0,
                max_value=500,
                step=10,
                help="チャンク間の重複文字数",
                disabled=not is_pdf_mode
            )
        with col3:
            use_azure_di = st.checkbox(
                "Azure DI 使用",
                value=False,
                help="Azure Document Intelligenceで高精度抽出",
                disabled=not is_pdf_mode
            )

    # Azure DI詳細オプション（常に表示）
    with st.expander("🔍 Azure DI 詳細オプション", expanded=False):
        if not is_pdf_mode:
            st.caption("💡 PDFモード + Azure DI使用時に有効になります")
        elif not use_azure_di:
            st.caption("💡 Azure DI使用にチェックを入れると有効になります")
        di_col1, di_col2, di_col3 = st.columns(3)
        with di_col1:
            extract_figures = st.checkbox(
                "図をテキスト化",
                value=True,
                help="Vision API で図を説明文に変換",
                disabled=not (is_pdf_mode and use_azure_di)
            )
        with di_col2:
            convert_tables = st.checkbox(
                "表をテキスト化",
                value=True,
                help="HTML表をLLMで文章に変換",
                disabled=not (is_pdf_mode and use_azure_di)
            )
        with di_col3:
            save_markdown = st.checkbox(
                "Markdownを保存",
                value=True,
                help="output/markdown/ に保存",
                disabled=not (is_pdf_mode and use_azure_di)
            )

    # テキストクリーニング設定
    with st.expander("🧹 テキストクリーニング", expanded=False):
        st.caption("PDFから抽出したテキストのノイズ除去設定")
        clean_level = st.select_slider(
            "クリーニングレベル",
            options=["off", "basic", "aggressive"],
            value="basic",
            format_func=lambda x: {"off": "なし", "basic": "基本", "aggressive": "積極的"}[x],
            help="off=クリーニングなし, basic=ページ番号・目次除去, aggressive=ヘッダ/フッタ自動検出"
        )

        if clean_level != "off":
            st.markdown("**適用される処理:**")
            checks = []
            if clean_level in ["basic", "aggressive"]:
                checks.extend([
                    "✅ Unicode正規化（NFKC）",
                    "✅ ページ番号除去",
                    "✅ 目次除去",
                    "✅ 改行修復（単語分断の修正）",
                    "✅ 数値・単位正規化",
                ])
            if clean_level == "aggressive":
                checks.extend([
                    "✅ ヘッダ/フッタ自動検出・除去",
                    "✅ 断片文フィルタ",
                ])
            st.markdown("  \n".join(checks))

    # データ拡張設定
    with st.expander("✨ データ拡張設定", expanded=False):
        augment = st.toggle("データ拡張を有効化", value=True)

        if augment:
            # 辞書/グラフベース（LLM不要）
            st.markdown("**辞書/グラフベース** (LLM不要)")
            col1, col2, col3 = st.columns(3)
            with col1:
                aug_dictionary = st.checkbox("📚 辞書定義", value=False)
            with col2:
                aug_generalized = st.checkbox("🔄 一般化", value=False)
            with col3:
                aug_graph = st.checkbox("🔗 グラフ関係性", value=False, help="※LLM使用")

            # LLMベース
            st.markdown("**LLMベース**")
            col1, col2, col3 = st.columns(3)
            with col1:
                aug_paraphrase = st.checkbox("💬 言い換え", value=True)
            with col2:
                aug_qa = st.checkbox("❓ Q&A", value=True)
            with col3:
                aug_summary = st.checkbox("📝 要約", value=False)

            col1, col2 = st.columns(2)
            with col1:
                aug_keywords = st.checkbox("🏷️ キーワード", value=False)
            with col2:
                aug_discussion = st.checkbox("💭 議論形式", value=False)

            # LLMベース使用時の注意
            if aug_paraphrase or aug_qa or aug_summary or aug_keywords or aug_discussion or aug_graph:
                st.caption("⚠️ LLMベース拡張には `AZURE_OPENAI_ENDPOINT` と `AZURE_OPENAI_API_KEY` が必要です")

            # 翻訳
            st.markdown("**翻訳**")
            col1, col2 = st.columns(2)
            with col1:
                aug_en = st.checkbox("🇺🇸 英語", value=False)
            with col2:
                aug_zh = st.checkbox("🇨🇳 中国語", value=False)

            # ファイルパス設定
            if aug_dictionary or aug_generalized or aug_graph:
                st.markdown("**ファイルパス**")
                col1, col2 = st.columns(2)
                with col1:
                    dict_file = st.text_input("辞書ファイル", value="data/dict/terms.json", key="dict_file")
                with col2:
                    graph_file = st.text_input("グラフファイル", value="data/graph/graph.json", key="graph_file")
            else:
                dict_file = "data/dict/terms.json"
                graph_file = "data/graph/graph.json"
        else:
            aug_paraphrase = aug_qa = aug_summary = False
            aug_keywords = aug_discussion = False
            aug_en = aug_zh = False
            aug_dictionary = aug_generalized = aug_graph = False
            dict_file = "data/dict/terms.json"
            graph_file = "data/graph/graph.json"

    # パッキング設定
    with st.expander("📦 パッキング設定", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            pack = st.toggle("パッキング有効化", value=True)
        with col2:
            if pack:
                max_seq_len = st.select_slider(
                    "最大シーケンス長",
                    options=[1024, 2048, 4096, 8192],
                    value=2048,
                    help="トークン数の上限"
                )
            else:
                max_seq_len = 2048
                st.caption("パッキング無効時は各チャンクがそのまま出力されます")

    # その他オプション
    with st.expander("⚡ その他オプション", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            no_shuffle = st.checkbox("🔀 シャッフルしない", value=False)
        with col2:
            keep_intermediate = st.checkbox("💾 中間ファイル保持", value=False)

        st.markdown("**トークン数計算**")
        use_tokenizer = st.checkbox("🔢 トークナイザーで正確に計算", value=False, help="指定しない場合は推定値")
        if use_tokenizer:
            tokenizer_name = st.text_input(
                "モデル名",
                value="llm-jp/llm-jp-3-13b",
                help="HuggingFaceのモデル名（例: llm-jp/llm-jp-3-13b, meta-llama/Llama-2-7b）"
            )
        else:
            tokenizer_name = None

    # 設定確定ボタン（通常のボタン）
    submitted = st.button("⚙️ 設定を確定", type="secondary", use_container_width=True)

    st.divider()

    # コマンド生成（設定確定後のみ表示）
    # session_stateで設定を保持
    if submitted:
        st.session_state["settings_confirmed"] = True
        st.session_state["cfg"] = {
            "input_mode": input_mode,
            "input_path": input_path,
            "output_file": output_file,
            "chunk_size": chunk_size if "PDF" in input_mode else 1500,
            "chunk_overlap": chunk_overlap if "PDF" in input_mode else 100,
            "use_azure_di": use_azure_di if "PDF" in input_mode else False,
            "extract_figures": extract_figures if "PDF" in input_mode and use_azure_di else False,
            "convert_tables": convert_tables if "PDF" in input_mode and use_azure_di else False,
            "save_markdown": save_markdown if "PDF" in input_mode and use_azure_di else False,
            "clean_level": clean_level,
            "augment": augment,
            "aug_paraphrase": aug_paraphrase if augment else False,
            "aug_qa": aug_qa if augment else False,
            "aug_summary": aug_summary if augment else False,
            "aug_keywords": aug_keywords if augment else False,
            "aug_discussion": aug_discussion if augment else False,
            "aug_en": aug_en if augment else False,
            "aug_zh": aug_zh if augment else False,
            "aug_dictionary": aug_dictionary if augment else False,
            "aug_generalized": aug_generalized if augment else False,
            "aug_graph": aug_graph if augment else False,
            "dict_file": dict_file if augment else "data/dict/terms.json",
            "graph_file": graph_file if augment else "data/graph/graph.json",
            "pack": pack,
            "max_seq_len": max_seq_len,
            "no_shuffle": no_shuffle,
            "keep_intermediate": keep_intermediate,
            "tokenizer_name": tokenizer_name,
        }
        st.rerun()

    # 設定確定済みの場合のみコマンド表示
    if not st.session_state.get("settings_confirmed"):
        st.info("👆 設定を選択して「設定を確定」ボタンを押してください")
        st.stop()

    # 保存された設定を使用
    cfg = st.session_state.get("cfg", {})
    input_mode = cfg.get("input_mode", input_mode)
    input_path = cfg.get("input_path", input_path if "input_path" in dir() else "data/input")
    output_file = cfg.get("output_file", output_file if "output_file" in dir() else "data/output/train.jsonl")
    chunk_size = cfg.get("chunk_size", 1500)
    chunk_overlap = cfg.get("chunk_overlap", 100)
    use_azure_di = cfg.get("use_azure_di", False)
    extract_figures = cfg.get("extract_figures", False)
    convert_tables = cfg.get("convert_tables", False)
    save_markdown = cfg.get("save_markdown", False)
    clean_level = cfg.get("clean_level", "basic")
    augment = cfg.get("augment", True)
    aug_paraphrase = cfg.get("aug_paraphrase", True)
    aug_qa = cfg.get("aug_qa", True)
    aug_summary = cfg.get("aug_summary", False)
    aug_keywords = cfg.get("aug_keywords", False)
    aug_discussion = cfg.get("aug_discussion", False)
    aug_en = cfg.get("aug_en", False)
    aug_zh = cfg.get("aug_zh", False)
    aug_dictionary = cfg.get("aug_dictionary", False)
    aug_generalized = cfg.get("aug_generalized", False)
    aug_graph = cfg.get("aug_graph", False)
    dict_file = cfg.get("dict_file", "data/dict/terms.json")
    graph_file = cfg.get("graph_file", "data/graph/graph.json")
    pack = cfg.get("pack", True)
    max_seq_len = cfg.get("max_seq_len", 2048)
    no_shuffle = cfg.get("no_shuffle", False)
    keep_intermediate = cfg.get("keep_intermediate", False)
    tokenizer_name = cfg.get("tokenizer_name", None)

    # コマンド生成
    cmd = [sys.executable, "scripts/batch_pipeline.py", input_path, "-o", output_file]
    # PDFモードのみチャンクオプションを追加
    if "PDF" in input_mode:
        cmd += ["--chunk-size", str(chunk_size)]
        cmd += ["--chunk-overlap", str(chunk_overlap)]
    if use_azure_di:
        cmd.append("--use-azure-di")
        if extract_figures:
            cmd.append("--extract-figures")
        if convert_tables:
            cmd.append("--convert-tables")
        if save_markdown:
            cmd.append("--save-markdown")
    if clean_level != "basic":  # basicはデフォルトなので省略
        cmd += ["--clean-level", clean_level]
    if augment:
        cmd.append("--augment")
        if aug_paraphrase:
            cmd.append("--aug-paraphrase")
        if aug_qa:
            cmd.append("--aug-qa")
        if aug_summary:
            cmd.append("--aug-summary")
        if aug_keywords:
            cmd.append("--aug-keywords")
        if aug_discussion:
            cmd.append("--aug-discussion")
        if aug_en:
            cmd.append("--aug-translation-en")
        if aug_zh:
            cmd.append("--aug-translation-zh")
        if aug_dictionary:
            cmd.append("--aug-dictionary")
        if aug_generalized:
            cmd.append("--aug-generalized")
        if aug_graph:
            cmd.append("--aug-graph")
        if dict_file != "data/dict/terms.json":
            cmd += ["--dict-file", dict_file]
        if graph_file != "data/graph/graph.json":
            cmd += ["--graph-file", graph_file]
    if pack:
        cmd.append("--pack")
        cmd += ["--max-seq-len", str(max_seq_len)]
    if no_shuffle:
        cmd.append("--no-shuffle")
    if keep_intermediate:
        cmd.append("--keep-intermediate")
    if tokenizer_name:
        cmd += ["--tokenizer", tokenizer_name]

    # コマンドプレビュー
    st.markdown("##### 🖥️ 実行コマンド")
    st.code(" ".join(cmd), language="bash")

    # 実行・リセットボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 設定変更", key="reset_settings", use_container_width=True):
            st.session_state["settings_confirmed"] = False
            st.rerun()
    with col2:
        run_button = st.button("🚀 パイプライン実行", key="run_batch", type="primary", use_container_width=True)

    if run_button:
        with st.status("パイプライン実行中...", expanded=True) as status:
            input_type = "PDF" if "PDF" in input_mode else "JSONL"
            st.write(f"📁 {input_type}ファイルを処理中...")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

            if result.returncode == 0:
                st.write(f"✅ {input_type}処理完了")
                st.write("✅ パッキング完了" if pack else "✅ 変換完了")
                st.write("✅ マージ完了")
                status.update(label="✨ パイプライン完了!", state="complete", expanded=False)
                st.toast("処理が完了しました!", icon="✅")
            else:
                status.update(label="❌ エラー発生", state="error")
                st.toast("エラーが発生しました", icon="❌")

        # ログ表示（statusブロックの外）
        if result.returncode == 0:
            # 統計情報を抽出して表示
            import re
            stdout = result.stdout or ""
            count_match = re.search(r'総データ数: ([\d,]+)件', stdout)
            chars_match = re.search(r'合計文字数: ([\d,]+)文字', stdout)
            tokens_match = re.search(r'(?:推定)?トークン数: ([\d,]+)トークン', stdout)
            is_estimated = "推定トークン数" in stdout

            if count_match and chars_match and tokens_match:
                st.markdown("##### 📊 生成データ統計")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 レコード数", count_match.group(1) + "件")
                with col2:
                    st.metric("📝 合計文字数", chars_match.group(1) + "字")
                with col3:
                    token_label = "🔢 推定トークン" if is_estimated else "🔢 トークン数"
                    st.metric(token_label, tokens_match.group(1))

            with st.expander("📋 実行ログ", expanded=False):
                st.text(stdout)
        else:
            st.error("処理中にエラーが発生しました")
            st.text(result.stderr or result.stdout)

# =============================================================================
# Tab 2: 個別スクリプト
# =============================================================================
with tab2:
    script_options = {
        "📚 辞書から用語定義": ("scripts/augment/expand_dictionary.py", "辞書ベース", False),
        "📖 専門用語に説明追加": ("scripts/augment/expand_elaboration.py", "辞書ベース", False),
        "🔄 専門用語→一般用語": ("scripts/augment/expand_generalized.py", "辞書ベース", False),
        "🏷️ キーワード抽出": ("scripts/augment/expand_keywords.py", "辞書ベース", False),
        "❓ Q&A生成": ("scripts/augment/expand_qa_difficult.py", "LLM使用", True),
        "🌐 英語翻訳": ("scripts/augment/expand_to_english.py", "LLM使用", True),
        "🔗 グラフ関係性": ("scripts/augment/expand_graph_relations.py", "LLM使用", True),
        "📦 パッキング": ("scripts/preprocess/pack_sequences.py", "ローカル処理", False),
    }

    # メトリクス
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📜 利用可能スクリプト", f"{len(script_options)}個")
    with m2:
        st.metric("🤖 LLM必要", "3個")
    with m3:
        st.metric("📚 辞書ベース", "4個")

    st.divider()

    selected_script = st.selectbox(
        "スクリプト選択",
        list(script_options.keys()),
        format_func=lambda x: x
    )
    script_path, script_type, needs_llm = script_options[selected_script]

    # スクリプト情報
    if needs_llm:
        st.warning(f"⚠️ このスクリプトはLLM APIを使用します（{script_type}）")
    else:
        st.info(f"ℹ️ {script_type}処理")

    st.divider()

    # スクリプト別の入力
    if "pack_sequences" in script_path:
        with st.expander("📦 パッキング設定", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                pack_input = st.text_input("入力JSONL", value="data/output/sample.jsonl", key="pack_input")
            with col2:
                pack_output = st.text_input("出力JSONL", value="data/output/packed.jsonl", key="pack_output")

            col1, col2 = st.columns(2)
            with col1:
                pack_max_seq = st.select_slider("最大シーケンス長", options=[1024, 2048, 4096, 8192], value=2048, key="pack_seq")
            with col2:
                pack_shuffle = st.checkbox("🔀 シャッフル", value=False, key="pack_shuf")

        cmd2 = [sys.executable, script_path, pack_input, "-o", pack_output, "--max-seq-len", str(pack_max_seq)]
        if pack_shuffle:
            cmd2.append("--shuffle")

    elif "graph_relations" in script_path:
        with st.expander("🔗 グラフ設定", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                graph_input = st.text_input("グラフJSON", value="data/graph/graph.json", key="graph_input")
            with col2:
                graph_output = st.text_input("出力JSONL", value="data/output/graph_relations.jsonl", key="graph_output")

            graph_limit = st.number_input("処理ノード数 (0=全て)", value=0, min_value=0, key="graph_limit")

        cmd2 = [sys.executable, script_path]
        if graph_input != "data/graph/graph.json":
            cmd2 += ["--input", graph_input]
        if graph_output != "data/output/graph_relations.jsonl":
            cmd2 += ["--output", graph_output]
        if graph_limit > 0:
            cmd2 += ["--limit", str(graph_limit)]
    else:
        st.caption("このスクリプトはデフォルト設定で実行されます")
        cmd2 = [sys.executable, script_path]

    # コマンドプレビュー
    st.markdown("##### 🖥️ 実行コマンド")
    st.code(" ".join(cmd2), language="bash")

    # 実行ボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_script = st.button("🚀 スクリプト実行", key="run_script", type="primary", use_container_width=True)

    if run_script:
        with st.status("スクリプト実行中...", expanded=True) as status:
            st.write(f"▶️ {selected_script} を実行中...")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", env=env)

            if result.returncode == 0:
                status.update(label="✨ 実行完了!", state="complete", expanded=False)
                st.toast("処理が完了しました!", icon="✅")
            else:
                status.update(label="❌ エラー発生", state="error")
                st.toast("エラーが発生しました", icon="❌")

        # ログ表示（statusブロックの外）
        if result.returncode == 0:
            with st.expander("📋 実行ログ", expanded=True):
                st.text(result.stdout)
        else:
            st.error("処理中にエラーが発生しました")
            st.text(result.stderr or result.stdout)

# =============================================================================
# Tab 3: ビューアー
# =============================================================================
with tab3:
    uploaded_file = st.file_uploader(
        "JSONLファイルをドラッグ&ドロップまたは選択",
        type=["jsonl", "json"],
        help="生成されたJSONLファイルをアップロードして内容を確認"
    )

    if uploaded_file is not None:
        import pandas as pd

        records = []
        for line_num, line in enumerate(uploaded_file, 1):
            try:
                record = json.loads(line)
                record["_line"] = line_num
                records.append(record)
            except json.JSONDecodeError:
                pass

        if records:
            df = pd.DataFrame(records)
            cols = ["_line"] + [c for c in df.columns if c != "_line"]
            df = df[cols]

            # メトリクス
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📊 総レコード数", len(records))
            with m2:
                if "augmentation_type" in df.columns:
                    st.metric("🏷️ 拡張タイプ", df["augmentation_type"].nunique())
                else:
                    st.metric("🏷️ 拡張タイプ", "-")
            with m3:
                avg_len = df["text"].str.len().mean() if "text" in df.columns else 0
                st.metric("📝 平均文字数", f"{avg_len:.0f}")
            with m4:
                st.metric("📁 ファイル名", uploaded_file.name[:15] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name)

            st.divider()

            # フィルター
            col1, col2 = st.columns(2)
            with col1:
                if "augmentation_type" in df.columns:
                    types = ["全て"] + list(df["augmentation_type"].dropna().unique())
                    selected_type = st.selectbox("🏷️ タイプでフィルター", types)
                    if selected_type != "全て":
                        df = df[df["augmentation_type"] == selected_type]
            with col2:
                search_text = st.text_input("🔍 テキスト検索", placeholder="キーワードを入力...")
                if search_text:
                    mask = df.apply(lambda row: search_text.lower() in str(row).lower(), axis=1)
                    df = df[mask]

            # フィルター結果
            st.caption(f"表示: **{len(df)}件** / {len(records)}件中")

            # データテーブル
            st.dataframe(
                df.head(100),
                use_container_width=True,
                height=350,
                column_config={
                    "_line": st.column_config.NumberColumn("行", width="small"),
                    "text": st.column_config.TextColumn("テキスト", width="large"),
                }
            )

            st.divider()

            # 詳細表示
            st.markdown("##### 🔎 レコード詳細")
            col1, col2 = st.columns([1, 4])
            with col1:
                selected_line = st.number_input(
                    "行番号",
                    min_value=1,
                    max_value=len(records),
                    value=1,
                    key="detail_line"
                )
            with col2:
                st.caption(f"行 {selected_line} / {len(records)}")

            selected_record = records[selected_line - 1]

            if "text" in selected_record:
                st.text_area(
                    "📝 テキスト内容",
                    selected_record["text"],
                    height=200,
                    key="detail_text"
                )

            other_fields = {k: v for k, v in selected_record.items() if k not in ["text", "_line"]}
            if other_fields:
                with st.expander("📋 その他のフィールド", expanded=True):
                    st.json(other_fields)
        else:
            st.warning("有効なJSONレコードが見つかりませんでした")
    else:
        # プレースホルダー
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            color: #888;
            margin: 2rem 0;
        ">
            <p style="font-size: 3rem; margin: 0;">📄</p>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">JSONLファイルをドラッグ&ドロップ</p>
            <p style="font-size: 0.9rem; color: #aaa;">または「Browse files」をクリック</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 対応形式"):
            st.code('{"text": "...", "augmentation_type": "...", ...}', language="json")
