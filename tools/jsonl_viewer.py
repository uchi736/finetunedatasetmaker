import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="JSONL Viewer", layout="wide")
st.title("JSONL Viewer")

uploaded_file = st.file_uploader("JSONLファイルを選択", type=["jsonl", "json"])

if uploaded_file is not None:
    records = []
    for line_num, line in enumerate(uploaded_file, 1):
        try:
            record = json.loads(line)
            # 行番号を追加
            record["_line"] = line_num
            records.append(record)
        except json.JSONDecodeError:
            st.warning(f"行{line_num}: 無効なJSONをスキップ")

    if records:
        df = pd.DataFrame(records)

        # _lineを先頭に移動
        cols = ["_line"] + [c for c in df.columns if c != "_line"]
        df = df[cols]

        # サイドバーでフィルタリング
        st.sidebar.header("フィルター")

        # augmentation_type でフィルタ（あれば）
        if "augmentation_type" in df.columns:
            types = ["全て"] + list(df["augmentation_type"].dropna().unique())
            selected_type = st.sidebar.selectbox("augmentation_type", types)
            if selected_type != "全て":
                df = df[df["augmentation_type"] == selected_type]

        # テキスト検索
        search_text = st.sidebar.text_input("テキスト検索")
        if search_text:
            mask = df.apply(lambda row: search_text.lower() in str(row).lower(), axis=1)
            df = df[mask]

        # 表示件数
        max_rows = st.sidebar.slider("表示件数", 10, 500, 100)

        st.write(f"**{len(df)}件** (フィルター後)")

        # テーブル表示
        st.dataframe(
            df.head(max_rows),
            use_container_width=True,
            height=600
        )

        # 詳細表示（行選択）
        st.subheader("詳細表示")
        selected_line = st.number_input("行番号を入力", min_value=1, max_value=len(records), value=1)
        if selected_line:
            selected_record = records[selected_line - 1]

            # テキストを大きく表示
            if "text" in selected_record:
                st.text_area("text", selected_record["text"], height=200)

            # その他のフィールド
            st.json({k: v for k, v in selected_record.items() if k != "text"})
    else:
        st.info("表示するデータがありません。")
else:
    st.info("JSONLファイルをアップロードしてください。")

    st.markdown("""
    ### 対応形式
    ```json
    {"text": "...", "augmentation_type": "...", ...}
    ```

    ### 使い方
    1. 「Browse files」でJSONLファイルを選択
    2. サイドバーでフィルタリング
    3. 行番号を入力して詳細表示
    """)
