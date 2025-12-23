import streamlit as st
import json
import pandas as pd

st.title("JSONL Viewer")

uploaded_file = st.file_uploader("Choose a JSONL file", type="jsonl")

if uploaded_file is not None:
    records = []
    for line in uploaded_file:
        try:
            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "original":
                data = record.get("data", {})
                meta = data.get("meta", {})
                records.append({
                    "type": record_type,
                    "id": data.get("id"),
                    "text": data.get("text"),
                    "source_file": meta.get("source_file"),
                    "chunk_id": meta.get("chunk_id"),
                    "section_path": meta.get("section_path"),
                })
            elif record_type in ["paraphrase", "summary", "qa"]:
                records.append({
                    "type": record_type,
                    "id": record.get("original_id"),
                    "text": record.get("text") or record.get("summary") or f"Q: {record.get('question')}\nA: {record.get('answer')}",
                    "source_file": None,
                    "chunk_id": None,
                    "section_path": None,
                })

        except json.JSONDecodeError:
            st.warning(f"Skipping invalid JSON line: {line.decode('utf-8', 'ignore')}")

    if records:
        df = pd.DataFrame(records)
        
        # 列の順序を調整
        column_order = ["id", "type", "text", "source_file", "chunk_id", "section_path"]
        df = df[column_order]

        st.dataframe(df)
    else:
        st.info("No data to display.")
