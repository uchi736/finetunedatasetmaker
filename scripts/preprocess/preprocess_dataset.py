# ========= データセット前処理スクリプト（GPU不要） =========
# このスクリプトはCPUのみで実行可能です

import os
from datasets import load_dataset
from transformers import AutoTokenizer
import warnings
from multiprocess import freeze_support

warnings.filterwarnings("ignore")

# ========= 設定 =========
model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.2"
output_dir = "./preprocessed_dolly_ja"

# ========= フォーマット関数 =========
def format_llama3_style(example):
    """Llama-3形式でフォーマット"""
    system_prompt = "あなたは誠実で有能な日本語のAIアシスタントです。ユーザーの質問に対して、正確で役立つ回答を提供してください。"
    
    instruction = example['instruction']
    input_context = example['input'] if 'input' in example and example['input'] else ""
    output = example['output']

    if input_context and input_context.strip():
        user_content = f"{instruction}\n\n背景情報:\n{input_context}"
    else:
        user_content = instruction
    
    # Llama-3形式
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
    
    return {"text": text}

def main():
    # ========= データセットのロード =========
    print("Dolly-JAデータセットをロード中...")
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")

    # データセットを学習用と検証用に分割
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"学習データ数: {len(train_dataset)}")
    print(f"検証データ数: {len(eval_dataset)}")
    print(f"カラム名: {train_dataset.column_names}")

    # ========= データセットのフォーマット実行 =========
    print("学習データをフォーマット中...")
    formatted_train = train_dataset.map(
        format_llama3_style,
        num_proc=4,  # CPUコア数に応じて調整
        remove_columns=train_dataset.column_names,
        desc="Formatting training data"
    )

    print("検証データをフォーマット中...")
    formatted_eval = eval_dataset.map(
        format_llama3_style,
        num_proc=4,  # CPUコア数に応じて調整
        remove_columns=eval_dataset.column_names,
        desc="Formatting evaluation data"
    )

    # ========= フォーマット済みデータセットの保存 =========
    print(f"\nフォーマット済みデータセットを {output_dir} に保存中...")
    os.makedirs(output_dir, exist_ok=True)

    # JSONL形式で保存
    train_output_path = os.path.join(output_dir, "train_formatted.jsonl")
    eval_output_path = os.path.join(output_dir, "eval_formatted.jsonl")
    
    formatted_train.to_json(train_output_path, orient="records", force_ascii=False, lines=True)
    formatted_eval.to_json(eval_output_path, orient="records", force_ascii=False, lines=True)

    print(f"学習データを {train_output_path} に保存しました。")
    print(f"検証データを {eval_output_path} に保存しました。")
    
    # トークナイザーも念のため保存
    print("\nトークナイザーを保存中...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))


    # ========= サンプル確認 =========
    print("\n=== フォーマット済みデータのサンプル ===")
    sample_text = formatted_train[0]["text"]
    print(sample_text)


    print("\nフォーマットが完了しました！")
    print(f"フォーマット済みデータは {output_dir} に保存されています。")

if __name__ == '__main__':
    freeze_support()
    main()
