#!/usr/bin/env python3
"""
統合報告書のような複雑なPDFドキュメントを高精度でテキスト化し、
LLM学習用データに変換する改良版パイプライン

主な特徴:
- レイアウト保持型の高精度テキスト抽出
- 見出し階層の自動認識
- 表の構造化抽出とMarkdown変換
- 図表のキャプション抽出と画像解説
- 2段組レイアウトの正確な読み順保持
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

import fitz  # PyMuPDF
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm
import pandas as pd
from datasketch import MinHash, MinHashLSH
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
from nltk.tokenize import sent_tokenize

# NLTKのpunktデータセットをダウンロード（初回のみ）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Vision API (Azure OpenAI) のインポート
try:
    from openai import AzureOpenAI
    import PIL.Image
    import base64
except ImportError:
    AzureOpenAI = None
    PIL = None
    print("Warning: openai or Pillow not found. Vision features will be disabled.")

# Azure Document Intelligence のインポート
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )
    AZURE_DI_AVAILABLE = True
except ImportError:
    AZURE_DI_AVAILABLE = False
    print("Warning: azure-ai-documentintelligence not found. Azure DI features will be disabled.")

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_japanese_text(text: str) -> str:
    """
    日本語テキストの改行・スペースアーティファクトをクリーンアップ

    Azure Document Intelligence等で抽出されたテキストには、
    「輸\n送機器向け」のような不要な改行や「削 減」のような
    不要なスペースが含まれることがある。これらを除去する。
    """
    # 1. Markdownソフト改行（2スペース+改行）をスペースに変換
    text = re.sub(r'  \n', ' ', text)

    # 2. 改行前後のスペースを除去
    text = re.sub(r' *\n *', '\n', text)

    # 3. 日本語文字間の不要なスペースを除去
    # 「削 減」→「削減」のようなケースに対応
    text = re.sub(r'[ ]+([ぁ-んァ-ヴー一-龠々〆〤])', r'\1', text)
    text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])[ ]+', r'\1', text)

    # 4. 日本語文字間の不要な改行を除去（複数改行も対応）
    # 「輸\n送」→「輸送」のようなケースに対応
    text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])\n+([ぁ-んァ-ヴー一-龠々〆〤])', r'\1\2', text)

    return text


# --- データ構造 ---
@dataclass
class ChunkData:
    """チャンクデータの構造体"""
    id: str
    text: str
    meta: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ExtractedBlock:
    """抽出されたテキストブロック"""
    text: str
    bbox: Tuple[float, float, float, float]
    block_type: str  # 'heading', 'paragraph', 'table', 'figure_caption'
    font_size: float = 0
    is_bold: bool = False
    page_num: int = 0
    column: int = 0  # 0: single column, 1: left, 2: right

# --- PDFレイアウト解析クラス ---
class LayoutAnalyzer:
    """PDFのレイアウトを解析して構造化データを抽出"""
    
    def __init__(self):
        self.heading_patterns = [
            r'^第\d+[章節]',
            r'^\d+\.',
            r'^[０-９]+\.',
            r'^■',
            r'^●',
            r'^▼',
            r'^【.+】$',
        ]
        
    def analyze_page_layout(self, page: fitz.Page) -> Dict[str, Any]:
        """ページのレイアウトを解析"""
        # ページの幅を取得
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # 中央のX座標（2段組の判定用）
        mid_x = page_width / 2
        
        # テキストブロックを取得
        blocks = page.get_text("dict", sort=True)
        
        # 段組判定
        is_two_column = self._detect_two_column_layout(blocks, mid_x)
        
        return {
            "page_width": page_width,
            "page_height": page_height,
            "mid_x": mid_x,
            "is_two_column": is_two_column,
            "blocks": blocks
        }
    
    def _detect_two_column_layout(self, blocks: Dict, mid_x: float) -> bool:
        """2段組レイアウトかどうかを判定"""
        left_blocks = 0
        right_blocks = 0
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # テキストブロック
                bbox = block.get("bbox", [0, 0, 0, 0])
                block_mid_x = (bbox[0] + bbox[2]) / 2
                
                if block_mid_x < mid_x * 0.9:
                    left_blocks += 1
                elif block_mid_x > mid_x * 1.1:
                    right_blocks += 1
        
        # 左右のブロック数が近い場合は2段組と判定
        return left_blocks > 5 and right_blocks > 5 and abs(left_blocks - right_blocks) < 10

# --- 高度なPDF処理クラス ---
class AdvancedPDFProcessor:
    """統合報告書向けの高度なPDF処理"""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        extract_tables: bool = True,
        extract_images: bool = True,
        use_azure_di: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.use_azure_di = use_azure_di

        self.layout_analyzer = LayoutAnalyzer()
        self._setup_vision_model()

        # Azure Document Intelligence の設定
        self.di_client = None
        if self.use_azure_di:
            self._setup_azure_di()

    def _setup_azure_di(self):
        """Azure Document Intelligence クライアントの設定"""
        if not AZURE_DI_AVAILABLE:
            logger.error("Azure Document Intelligence library not available.")
            return

        endpoint = os.getenv('AZURE_DI_ENDPOINT')
        api_key = os.getenv('AZURE_DI_API_KEY')

        if not endpoint or not api_key:
            logger.error("AZURE_DI_ENDPOINT and AZURE_DI_API_KEY must be set in environment variables.")
            return

        try:
            self.di_client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key)
            )
            logger.info("Azure Document Intelligence client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Document Intelligence client: {e}")

    def _setup_vision_model(self):
        """Azure OpenAI Vision APIの設定"""
        self.vision_client = None
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')

        if endpoint and api_key and AzureOpenAI:
            try:
                self.vision_client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-02-15-preview"
                )
                logger.info("Azure OpenAI Vision model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI model: {e}")
    
    def process_pdf(self, pdf_path: str) -> List[ChunkData]:
        """PDFを処理してチャンクデータを生成"""
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # 1. Markdown抽出（Azure DI または pymupdf）
            if self.use_azure_di and self.di_client:
                markdown_text = self._extract_with_azure_di(pdf_path)
            else:
                markdown_text = self._extract_structured_markdown(pdf_path)

            # ヘッダーが正しく分離されるように、ヘッダーの前に空行を挿入
            markdown_text = re.sub(r'\n(#+)', r'\n\n\1', markdown_text)

            # 2. 画像の抽出と解説生成（pymupdfの場合のみ）
            if not self.use_azure_di and self.extract_images and self.vision_client:
                markdown_text = self._process_images(pdf_path, markdown_text)

            # 3. Markdownのクリーンアップ
            markdown_text = self._clean_markdown(markdown_text)

            # 4. 日本語テキストのクリーンアップ
            markdown_text = clean_japanese_text(markdown_text)

            # 5. チャンク分割
            chunks = self._create_chunks(markdown_text, pdf_path.name)

            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []

    def _extract_with_azure_di(self, pdf_path: Path) -> str:
        """Azure Document Intelligence を使用してMarkdownを抽出"""
        logger.info(f"Extracting with Azure Document Intelligence: {pdf_path}")

        try:
            with open(pdf_path, "rb") as f:
                file_content = f.read()

            # Azure Document Intelligence で解析
            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=file_content),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )

            result = poller.result()

            if result.content:
                logger.info(f"Azure DI extracted {len(result.content)} characters")
                return result.content
            else:
                logger.warning("Azure DI returned no content, falling back to pymupdf")
                return self._extract_structured_markdown(pdf_path)

        except Exception as e:
            logger.error(f"Azure DI extraction failed: {e}, falling back to pymupdf")
            return self._extract_structured_markdown(pdf_path)
    
    def _extract_structured_markdown(self, pdf_path: Path) -> str:
        """構造化されたMarkdownを抽出"""
        try:
            doc = fitz.open(pdf_path)
            
            # pymupdf4llmの高度な設定でMarkdown抽出
            md_text_list = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                write_images=self.extract_images,
                image_path="./extracted_images",
                image_format="png",
                dpi=150,
                margins=(0, 0, 0, 0)
            )
            
            # リストの各要素が辞書か文字列かを確認し、適切にテキストを抽出
            processed_texts = []
            for item in md_text_list:
                if isinstance(item, dict) and 'text' in item:
                    processed_texts.append(item['text'])
                elif isinstance(item, str):
                    processed_texts.append(item)
            md_text = "\n\n".join(processed_texts)
            
            # レイアウト解析による追加処理
            if self.extract_tables:
                md_text = self._enhance_with_table_extraction(doc, md_text)
            
            doc.close()
            
            # 階層構造の正規化
            md_text = self._normalize_headings(md_text)
            
            return md_text
            
        except Exception as e:
            logger.error(f"Error extracting markdown: {e}")
            # フォールバック: 基本的なテキスト抽出
            return self._fallback_text_extraction(pdf_path)
    
    def _enhance_with_table_extraction(self, doc: fitz.Document, markdown_text: str) -> str:
        """表の構造化抽出で強化"""
        for page_num, page in enumerate(doc):
            # PyMuPDFの表検出機能を使用
            tables = page.find_tables()
            
            for table_num, table in enumerate(tables):
                try:
                    # 表をpandasのDataFrameに変換
                    df = table.to_pandas()
                    
                    # Markdown形式に変換
                    table_md = df.to_markdown(index=False)
                    
                    # 元のMarkdownテキストに表を挿入
                    table_placeholder = f"[Table {page_num+1}-{table_num+1}]"
                    if table_placeholder in markdown_text:
                        markdown_text = markdown_text.replace(table_placeholder, table_md)
                    else:
                        # ページマーカーの後に追加
                        page_marker = f"<!-- Page {page_num+1} -->"
                        if page_marker in markdown_text:
                            markdown_text = markdown_text.replace(
                                page_marker,
                                f"{page_marker}\n\n### 表 {page_num+1}-{table_num+1}\n{table_md}\n"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract table on page {page_num+1}: {e}")
        
        return markdown_text
    
    def _normalize_headings(self, text: str) -> str:
        """見出しレベルの正規化"""
        # 大見出しパターン
        major_patterns = [
            (r'^(第\d+[章])\s*(.+)$', r'## \1 \2'),
            (r'^(\d+\.)\s*(.+)$', r'## \1 \2'),
            (r'^■\s*(.+)$', r'## \1'),
            (r'^【(.+)】$', r'## \1'),
        ]
        
        # 中見出しパターン
        minor_patterns = [
            (r'^(第\d+[節])\s*(.+)$', r'### \1 \2'),
            (r'^(\d+\.\d+)\s*(.+)$', r'### \1 \2'),
            (r'^●\s*(.+)$', r'### \1'),
            (r'^▼\s*(.+)$', r'### \1'),
        ]
        
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                normalized_lines.append('')
                continue
            
            # 大見出しの変換
            for pattern, replacement in major_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    line = re.sub(pattern, replacement, line, flags=re.MULTILINE)
                    break
            
            # 中見出しの変換
            for pattern, replacement in minor_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    line = re.sub(pattern, replacement, line, flags=re.MULTILINE)
                    break
            
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _process_images(self, pdf_path: Path, markdown_text: str) -> str:
        """Markdown内の画像を処理し、解説を生成または不要な画像を削除"""
        
        # Markdown内の画像タグをすべて見つける
        image_pattern = r"!\[(.*?)\]\((.*?)\)"
        matches = list(re.finditer(image_pattern, markdown_text))
        
        # 後ろから処理していくことで、文字列置換によるインデックスのズレを防ぐ
        for match in reversed(matches):
            alt_text = match.group(1)
            img_path_str = match.group(2)
            
            # URLやデータURIはスキップ
            if img_path_str.startswith(('http', 'data:')):
                continue

            img_path = Path(img_path_str)
            
            # ローカルファイルでない場合はスキップ
            if not os.path.exists(img_path):
                continue
                
            try:
                with PIL.Image.open(img_path) as image:
                    width, height = image.size
                
                # 画像サイズが小さすぎる場合はタグごと削除
                if width < 50 or height < 50:
                    start, end = match.span()
                    # 前後の改行も削除
                    pre_text = markdown_text[:start]
                    post_text = markdown_text[end:]
                    if pre_text.endswith('\n'):
                        pre_text = pre_text.rstrip('\n')
                    if post_text.startswith('\n'):
                        post_text = post_text.lstrip('\n')
                    markdown_text = pre_text + post_text
                    
                    # 小さい画像ファイルも削除
                    try:
                        os.remove(img_path)
                    except OSError as e:
                        logger.warning(f"Could not remove small image file {img_path}: {e}")
                    continue

                # Geminiで画像の内容を分析
                analysis_result = self._generate_image_description(img_path)
                
                if analysis_result.startswith("TEXT_ONLY:"):
                    # 画像がテキストだった場合
                    extracted_text = analysis_result.replace("TEXT_ONLY:", "").strip()
                    
                    # 元の画像タグを抽出したテキストで置換
                    start, end = match.span()
                    # 抽出したテキストの前後に改行を追加して、Markdownのレンダリングを安定させる
                    markdown_text = markdown_text[:start] + f"\n\n{extracted_text}\n\n" + markdown_text[end:]
                    
                    # テキストとして処理したので画像ファイルは不要
                    try:
                        os.remove(img_path)
                    except OSError as e:
                        logger.warning(f"Could not remove text-only image file {img_path}: {e}")
                else:
                    # 通常の画像だった場合
                    description = analysis_result
                    # 新しいMarkdownブロックを作成
                    new_block = (
                        f"\n\n### 図: {alt_text if alt_text else img_path.name}\n"
                        f"![{alt_text}]({img_path_str})\n"
                        f"**説明**: {description}\n\n"
                    )
                    
                    # 元のタグを新しいブロックで置換
                    start, end = match.span()
                    markdown_text = markdown_text[:start] + new_block + markdown_text[end:]

            except Exception as e:
                logger.warning(f"Error processing image at {img_path}: {e}")
                
        return markdown_text
    
    def _generate_image_description(self, img_path: Path) -> str:
        """Azure OpenAI Vision APIを使用して画像の説明を生成"""
        if not self.vision_client:
            return "画像の説明は利用できません。"

        try:
            # 画像をbase64エンコード
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # 拡張子からMIMEタイプを判定
            ext = img_path.suffix.lower()
            mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(ext, "image/png")

            prompt = """この画像を分析してください。
もし画像が主に枠で囲まれたテキストである場合、"TEXT_ONLY:"という接頭辞に続けて、画像内のテキストを全て書き起こしてください。
それ以外の場合は、この画像の内容を統合報告書に掲載するキャプションとして、以下の形式で簡潔に日本語で説明してください。

- **種類**: 画像の種類を特定します（例: 製品写真, ウェブサイトのスクリーンショット, ロゴ, グラフ, 表）。
- **内容**: 画像に写っている主要な要素や情報を客観的に記述します。
- **示唆**: (任意) この画像がビジネスや経営において持つ可能性のある意味や示唆を簡潔に述べます。

説明は客観的かつ簡潔にまとめてください。"""

            response = self.vision_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            result_text = response.choices[0].message.content.strip()

            # テキストのみの場合は、そのまま返す
            if result_text.startswith("TEXT_ONLY:"):
                return result_text

            # 不要な定型文を削除
            unwanted_phrases = [
                "この画像は図表ではありません。",
                "これは図表ではありません。",
                "図表ではありません。",
            ]
            for phrase in unwanted_phrases:
                result_text = result_text.replace(phrase, "")

            return re.sub(r'\n{3,}', '\n\n', result_text).strip()

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return "画像の説明の生成に失敗しました。"

    def _clean_markdown(self, text: str) -> str:
        """不要なMarkdownの構文をクリーンアップ"""
        # 1. コードブロック(```...```)を解除
        text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)
        
        # 2. インラインコード(`...`)を解除
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 3. 段落間の改行を保護
        # 2つ以上の連続した改行を一時的なプレースホルダーに置換
        placeholder = "__PARAGRAPH_BREAK__"
        text = re.sub(r'\n{2,}', placeholder, text)
        
        # 4. 文中の不要な改行(単一の改行)をスペースに置換
        text = text.replace('\n', ' ')
        
        # 5. 保護した段落間の改行を復元
        text = text.replace(placeholder, '\n\n')

        # 6. 誤って結合された見出しを分離する
        # 例: "#見出し1 ##見出し2" -> "#見出し1\n##見出し2"
        text = re.sub(r'(#+) (.*?) (#+)', r'\1 \2\n\3', text)
        
        return text.strip()
    
    def _create_chunks(self, markdown_text: str, source_file: str) -> List[ChunkData]:
        """Markdownテキストをヘッダーに基づいてチャンク分割し、階層的なメタデータを付与"""
        
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]

        # 1. Markdownヘッダーで大まかに分割
        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_docs = md_header_splitter.split_text(markdown_text)

        # 2. 各ヘッダーセクションをさらにサイズで分割
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = size_splitter.split_documents(md_docs)

        # 3. 最終的なChunkDataリストを作成
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_text = doc.page_content.strip()
            if not chunk_text:
                continue

            # メタデータからセクションパスを構築
            headers = []
            if "H1" in doc.metadata: headers.append(doc.metadata["H1"])
            if "H2" in doc.metadata: headers.append(doc.metadata["H2"])
            if "H3" in doc.metadata: headers.append(doc.metadata["H3"])
            if "H4" in doc.metadata: headers.append(doc.metadata["H4"])
            section_path = ' › '.join(headers)

            # ページ番号の抽出
            page_numbers = self._extract_page_numbers(chunk_text)
            
            chunk = ChunkData(
                id=f"{Path(source_file).stem}_c{i+1}",
                text=chunk_text,
                meta={
                    "source_file": source_file,
                    "chunk_id": i + 1,
                    "section_path": section_path,
                    "page_start": page_numbers[0] if page_numbers else None,
                    "page_end": page_numbers[-1] if page_numbers else None,
                    "hierarchical_level": len(headers)
                }
            )
            chunks.append(chunk)
            
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _extract_page_numbers(self, text: str) -> List[int]:
        """テキストからページ番号を抽出"""
        page_pattern = r'<!-- Page (\d+) -->'
        matches = re.findall(page_pattern, text)
        return [int(m) for m in matches]
    
    def _fallback_text_extraction(self, pdf_path: Path) -> str:
        """フォールバック用の基本的なテキスト抽出"""
        text_parts = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text", sort=True)
                if text.strip():
                    text_parts.append(f"<!-- Page {page_num+1} -->\n{text}")
            
            doc.close()
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return ""

# --- データ拡張クラス ---
class DataAugmenter:
    """データ拡張機能を提供するクラス（Azure OpenAI使用）"""
    def __init__(self):
        self.client = None
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')

        if endpoint and api_key and AzureOpenAI:
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-02-15-preview"
                )
                logger.info("DataAugmenter: Azure OpenAI model initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI for DataAugmenter: {e}")

    def _call_llm(self, prompt: str) -> str:
        """LLM呼び出しの共通処理"""
        if not self.client:
            return ""
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()

    def generate_paraphrase(self, text: str) -> str:
        """言い換え生成"""
        if not self.client or not text.strip():
            return ""
        prompt = f"以下の文章を、元の意味を完全に保持したまま、異なる表現で言い換えてください。言い換えた後の文章のみを出力してください。\n\n# 元の文章:\n{text}"
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
            return ""

    def generate_qa(self, context: str) -> Dict[str, str]:
        """QAペア生成"""
        if not self.client or not context.strip():
            return {}
        prompt = f"""以下の文章に基づいて、質の高い質問と回答を1ペア生成してください。
出力は以下のJSON形式で:
```json
{{"question": "質問", "answer": "回答"}}
```

# 文章:
{context}"""
        try:
            response_text = self._call_llm(prompt)
            json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")
            return {}

    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """要約生成"""
        if not self.client or not text.strip():
            return ""
        prompt = f"以下の文章を{max_length}文字程度で簡潔に要約してください。\n\n# 文章:\n{text}"
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return ""

    def generate_keywords(self, text: str) -> List[Dict[str, str]]:
        """キーワード抽出"""
        if not self.client or not text.strip():
            return []
        prompt = f"""以下の文章から重要なキーワードを5つ抽出し、説明を付けてください。
出力は以下のJSON形式で:
```json
[{{"keyword": "キーワード", "description": "説明"}}]
```

# 文章:
{text}"""
        try:
            response_text = self._call_llm(prompt)
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"Keyword generation failed: {e}")
            return []

    def generate_elaboration(self, text: str) -> str:
        """詳細化テキスト生成"""
        if not self.client or not text.strip():
            return ""
        prompt = f"以下の文章を、具体例や背景情報を補足しながらより詳細に説明してください。\n\n# 元の文章:\n{text}"
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Elaboration generation failed: {e}")
            return ""

    def generate_translation(self, text: str, lang: str = "英語") -> str:
        """翻訳"""
        if not self.client or not text.strip():
            return ""
        prompt = f"以下の日本語の文章を自然な{lang}に翻訳してください。翻訳後の文章のみを出力。\n\n# 元の文章:\n{text}"
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Translation to {lang} failed: {e}")
            return ""

    def generate_specialized_qa(self, context: str, qa_type: str) -> Dict[str, str]:
        """特定観点のQA生成"""
        if not self.client or not context.strip():
            return {}
        type_desc = {"methods": "手法・技術", "people": "人物名・組織名", "numbers": "数値・統計データ"}
        prompt = f"""以下の文章から「{type_desc.get(qa_type, '一般')}」に焦点を当てたQAを1ペア生成。
出力はJSON形式: {{"question": "質問", "answer": "回答"}}

# 文章:
{context}"""
        try:
            response_text = self._call_llm(prompt)
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {}
        except Exception as e:
            logger.warning(f"Specialized QA ({qa_type}) failed: {e}")
            return {}

    def generate_discussion(self, text: str, turns: int = 3) -> str:
        """議論形式の対話生成"""
        if not self.client or not text.strip():
            return ""
        prompt = f"""以下の文章について専門家同士の議論を{turns}往復で生成してください。
形式: 質問者: (内容) / 回答者: (内容)

# 元の文章:
{text}"""
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Discussion generation failed: {e}")
            return ""

# --- 品質管理クラス ---
class QualityController:
    """データの品質を管理するクラス"""
    def __init__(self, perplexity_threshold: float = 30000.0, jaccard_threshold: float = 0.85):
        self.perplexity_threshold = perplexity_threshold
        self.jaccard_threshold = jaccard_threshold
        
        # Perplexity計算モデルの初期化
        try:
            model_name = "rinna/japanese-gpt2-small"
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ppl_model.to(self.device)
            logger.info(f"Perplexity model '{model_name}' loaded on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
            self.ppl_model = None

        # MinHash LSHの初期化
        self.lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=128)
        self.minhashes = {}

    def filter_by_perplexity(self, text: str) -> bool:
        """パープレキシティによる低流暢性テキストの除外。流暢ならTrueを返す"""
        if not self.ppl_model or not text.strip():
            return True # モデルがない場合やテキストが空の場合は除外しない
        try:
            inputs = self.ppl_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.ppl_model(**inputs, labels=inputs["input_ids"])
            
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
            is_fluent = perplexity.item() < self.perplexity_threshold
            if not is_fluent:
                logger.info(f"Filtered out by perplexity ({perplexity.item():.2f}): {text[:80]}...")
            return is_fluent
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return True # エラー時は除外しない

    def check_duplicate(self, chunk_id: str, text: str) -> bool:
        """MinHashによる重複チェック。重複ならTrueを返す"""
        if not text.strip():
            return True # 空のテキストは重複とみなす
            
        m = MinHash(num_perm=128)
        # テキストを単語に分割してMinHashを計算
        words = text.split()
        if not words:
            return True
        for d in words:
            m.update(d.encode('utf8'))
        
        # LSHで類似する候補を検索
        candidates = self.lsh.query(m)
        if any(m.jaccard(self.minhashes[c_id]) > self.jaccard_threshold for c_id in candidates):
            logger.info(f"Duplicate detected for chunk {chunk_id} with candidates {candidates}")
            return True # 重複あり
        
        # 新しいMinHashを登録
        self.lsh.insert(chunk_id, m)
        self.minhashes[chunk_id] = m
        return False # 重複なし

    def check_safety(self, text: str) -> bool:
        """個人情報などの安全性チェック。安全ならTrueを返す"""
        # 簡単な正規表現によるチェック
        patterns = {
            "phone": r'\b\d{2,4}-\d{2,4}-\d{4}\b',
            "email": r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "my_number": r'\b\d{12}\b',
        }
        
        for key, pattern in patterns.items():
            if re.search(pattern, text):
                logger.warning(f"Potential PII detected ({key}): {text[:100]}...")
                return False # 安全でない
        
        return True # 安全

# --- メイン処理 ---
def save_to_jsonl(data: List[Dict], output_path: str):
    """データをJSONL形式で保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    logger.info(f"Saved {len(data)} items to {output_path}")

def main():
    """メイン処理"""
    import argparse

    # config/.envから環境変数を読み込み
    load_dotenv('config/.env')

    parser = argparse.ArgumentParser(
        description="統合報告書などの複雑なPDFを高精度でLLM学習用データに変換",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 入出力設定
    parser.add_argument("pdf_path", help="入力PDFファイルのパス")
    parser.add_argument("--output", "-o", help="出力JSONLファイルのパス")

    # PDF抽出設定
    parser.add_argument("--chunk-size", type=int, default=1500, help="チャンクサイズ（文字数）")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="チャンク間のオーバーラップ")
    parser.add_argument("--no-extract-tables", dest="extract_tables", action="store_false", help="表を抽出しない")
    parser.add_argument("--no-extract-images", dest="extract_images", action="store_false", help="画像を抽出しない")
    parser.add_argument("--use-azure-di", action="store_true", help="Azure Document Intelligence を使用してPDFを抽出")

    # 品質管理設定
    parser.add_argument("--enable-quality-control", action="store_true", help="品質管理を有効にする")
    parser.add_argument("--perplexity-threshold", type=float, default=30000.0, help="パープレキシティの閾値")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85, help="Jaccard類似度の閾値（重複判定）")

    # データ拡張設定
    parser.add_argument("--enable-augmentation", action="store_true", help="データ拡張を有効にする")
    parser.add_argument("--aug-paraphrase", action="store_true", help="言い換え生成を有効にする")
    parser.add_argument("--aug-qa", action="store_true", help="QA生成を有効にする")
    parser.add_argument("--aug-summary", action="store_true", help="要約生成を有効にする")
    parser.add_argument("--aug-keywords", action="store_true", help="キーワード抽出と説明の生成を有効にする")
    parser.add_argument("--aug-elaboration", action="store_true", help="詳細化の生成を有効にする")
    parser.add_argument("--aug-translation-en", action="store_true", help="英語への翻訳を有効にする")
    parser.add_argument("--aug-translation-zh", action="store_true", help="中国語への翻訳を有効にする")
    parser.add_argument("--aug-qa-methods", action="store_true", help="手法に関するQA生成を有効にする")
    parser.add_argument("--aug-qa-people", action="store_true", help="人物に関するQA生成を有効にする")
    parser.add_argument("--aug-discussion", action="store_true", help="議論形式の対話生成を有効にする")

    args = parser.parse_args()

    # --- 初期化 ---
    if args.output:
        output_path = args.output
    else:
        pdf_path = Path(args.pdf_path)
        output_path = f"{pdf_path.stem}_dataset.jsonl"

    processor = AdvancedPDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extract_tables=args.extract_tables,
        extract_images=args.extract_images,
        use_azure_di=args.use_azure_di
    )

    quality_controller = QualityController(
        perplexity_threshold=args.perplexity_threshold,
        jaccard_threshold=args.jaccard_threshold
    ) if args.enable_quality_control else None

    augmenter = DataAugmenter() if args.enable_augmentation else None

    # --- パイプライン実行 ---
    
    # 1. PDFからチャンクを抽出
    initial_chunks = processor.process_pdf(args.pdf_path)
    if not initial_chunks:
        print("\nPDFからのチャンク抽出に失敗しました。処理を終了します。")
        return

    final_dataset = []
    
    # 2. 品質管理とデータ拡張
    for chunk in tqdm(initial_chunks, desc="Processing Chunks"):
        chunk_dict = chunk.to_dict()
        
        # 品質チェック
        if quality_controller:
            if not quality_controller.check_safety(chunk.text):
                continue
            if quality_controller.check_duplicate(chunk.id, chunk.text):
                continue
            if not quality_controller.filter_by_perplexity(chunk.text):
                continue
        
        # オリジナルのチャンクを追加（シンプルなtext形式）
        final_dataset.append({"text": chunk.text})

        # データ拡張
        if augmenter:
            if args.aug_paraphrase:
                paraphrase = augmenter.generate_paraphrase(chunk.text)
                if paraphrase:
                    final_dataset.append({"text": paraphrase})

            if args.aug_qa:
                qa_pair = augmenter.generate_qa(chunk.text)
                if qa_pair and qa_pair.get("question") and qa_pair.get("answer"):
                    # QAペアをテキスト形式に変換
                    qa_text = f"質問: {qa_pair['question']}\n回答: {qa_pair['answer']}"
                    final_dataset.append({"text": qa_text})

            if args.aug_summary:
                summary = augmenter.generate_summary(chunk.text)
                if summary:
                    final_dataset.append({"text": summary})

            if args.aug_keywords:
                keywords = augmenter.generate_keywords(chunk.text)
                if keywords:
                    # キーワードをテキスト形式に変換
                    kw_text = "\n".join([f"{kw['keyword']}: {kw['description']}" for kw in keywords if kw.get('keyword')])
                    if kw_text:
                        final_dataset.append({"text": kw_text})

            if args.aug_elaboration:
                elaboration = augmenter.generate_elaboration(chunk.text)
                if elaboration:
                    final_dataset.append({"text": elaboration})

            if args.aug_translation_en:
                translation = augmenter.generate_translation(chunk.text, lang="英語")
                if translation:
                    final_dataset.append({"text": translation})

            if args.aug_translation_zh:
                translation = augmenter.generate_translation(chunk.text, lang="中国語")
                if translation:
                    final_dataset.append({"text": translation})

            if args.aug_qa_methods:
                qa_pair = augmenter.generate_specialized_qa(chunk.text, qa_type="methods")
                if qa_pair and qa_pair.get("question") and qa_pair.get("answer"):
                    qa_text = f"質問: {qa_pair['question']}\n回答: {qa_pair['answer']}"
                    final_dataset.append({"text": qa_text})

            if args.aug_qa_people:
                qa_pair = augmenter.generate_specialized_qa(chunk.text, qa_type="people")
                if qa_pair and qa_pair.get("question") and qa_pair.get("answer"):
                    qa_text = f"質問: {qa_pair['question']}\n回答: {qa_pair['answer']}"
                    final_dataset.append({"text": qa_text})

            if args.aug_discussion:
                discussion = augmenter.generate_discussion(chunk.text)
                if discussion:
                    final_dataset.append({"text": discussion})

    # 3. 結果を保存
    if final_dataset:
        save_to_jsonl(final_dataset, output_path)
        print(f"\n処理完了!")
        print(f"  抽出チャンク数: {len(initial_chunks)}")
        print(f"  出力データ数: {len(final_dataset)}")
        print(f"  出力ファイル: {output_path}")
    else:
        print("\n処理に失敗しました。有効なデータが生成されませんでした。")

if __name__ == "__main__":
    main()
