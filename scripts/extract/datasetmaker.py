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

# テキストクリーニング
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from preprocess.clean_text import TextCleaner
    TEXT_CLEANER_AVAILABLE = True
except ImportError:
    TEXT_CLEANER_AVAILABLE = False
    TextCleaner = None

# プロンプト定義
from prompts import get_user_prompt, get_specialized_qa_prompt, SPECIALIZED_QA_TYPES

# NLTKのpunktデータセットをダウンロード（初回のみ）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Vision API (Azure OpenAI) のインポート
try:
    from openai import AzureOpenAI, AsyncAzureOpenAI
    import PIL.Image
    import base64
except ImportError:
    AzureOpenAI = None
    AsyncAzureOpenAI = None
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
        clean_level: str = "basic",  # off, basic, aggressive
        extract_figures: bool = False,  # Azure DI使用時に図をVision APIでテキスト化
        convert_tables: bool = False,   # HTML形式の表をLLMでテキスト化
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.use_azure_di = use_azure_di
        self.clean_level = clean_level
        self.extract_figures = extract_figures
        self.convert_tables = convert_tables

        self.layout_analyzer = LayoutAnalyzer()
        self._setup_vision_model()

        # テキストクリーナーの設定
        self.text_cleaner = None
        if TEXT_CLEANER_AVAILABLE and clean_level != "off":
            self.text_cleaner = TextCleaner(level=clean_level)
            logger.info(f"TextCleaner initialized with level: {clean_level}")

        # Azure Document Intelligence の設定
        self.di_client = None
        if self.use_azure_di:
            self._setup_azure_di()

        # 最後に処理したMarkdownを保持（Azure DI使用時に保存するため）
        self._last_markdown = None
        # 最後に処理した図情報を保持
        self._last_figures = None
        self._last_operation_id = None

    def _setup_azure_di(self):
        """Azure Document Intelligence クライアントの設定"""
        if not AZURE_DI_AVAILABLE:
            raise RuntimeError(
                "Azure Document Intelligence ライブラリがインストールされていません。\n"
                "→ pip install azure-ai-documentintelligence"
            )

        endpoint = os.getenv('AZURE_DI_ENDPOINT')
        api_key = os.getenv('AZURE_DI_API_KEY')

        if not endpoint:
            raise RuntimeError(
                "AZURE_DI_ENDPOINT が設定されていません。\n"
                "→ config/.env に AZURE_DI_ENDPOINT=https://xxx.cognitiveservices.azure.com/ を追加"
            )

        if not api_key:
            raise RuntimeError(
                "AZURE_DI_API_KEY が設定されていません。\n"
                "→ config/.env に AZURE_DI_API_KEY=xxx を追加"
            )

        try:
            self.di_client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key)
            )
            logger.info("Azure Document Intelligence client initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Azure DI クライアント初期化失敗: {e}")

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

            # 抽出結果チェック
            if not markdown_text or not markdown_text.strip():
                raise RuntimeError(
                    f"テキストが抽出できませんでした: {pdf_path}\n"
                    "→ PDFが画像のみの場合は --use-azure-di オプションを使用してください"
                )

            extracted_len = len(markdown_text)
            logger.info(f"Extracted {extracted_len} characters from {pdf_path.name}")

            # ヘッダーが正しく分離されるように、ヘッダーの前に空行を挿入
            markdown_text = re.sub(r'\n(#+)', r'\n\n\1', markdown_text)

            # 2. 画像の抽出と解説生成（pymupdfの場合のみ）
            if not self.use_azure_di and self.extract_images and self.vision_client:
                markdown_text = self._process_images(pdf_path, markdown_text)

            # 2.5. Azure DI使用時の図表処理
            if self.use_azure_di:
                # 図のVision API処理
                if self.extract_figures and self._last_figures:
                    markdown_text = self._process_azure_di_figures(markdown_text)

                # HTMLテーブルのテキスト化
                if self.convert_tables:
                    markdown_text = self._convert_html_tables_to_text(markdown_text)

            # 3. Markdownのクリーンアップ
            markdown_text = self._clean_markdown(markdown_text)

            # 4. 日本語テキストのクリーンアップ
            markdown_text = clean_japanese_text(markdown_text)

            # 5. 高度なテキストクリーニング（ページ番号、ヘッダ/フッタ、目次除去など）
            before_clean_len = len(markdown_text)
            if self.text_cleaner:
                clean_result = self.text_cleaner.clean(markdown_text)
                markdown_text = clean_result.text
                if clean_result.stats:
                    logger.info(f"TextCleaner: {clean_result.stats.get('reduction_rate', 0):.1%} reduction")

            # クリーニング後チェック
            if not markdown_text or not markdown_text.strip():
                raise RuntimeError(
                    f"クリーニング処理で全テキストが削除されました: {pdf_path}\n"
                    f"→ 元のテキスト: {before_clean_len}文字 → クリーニング後: 0文字\n"
                    "→ --clean-level off で再試行してください"
                )

            # 処理後のMarkdownを保持（後で取得可能）
            self._last_markdown = markdown_text

            # 6. チャンク分割
            chunks = self._create_chunks(markdown_text, pdf_path.name)

            if not chunks:
                raise RuntimeError(
                    f"チャンク分割で0件になりました: {pdf_path}\n"
                    f"→ テキスト長: {len(markdown_text)}文字"
                )

            return chunks

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise RuntimeError(f"PDF処理エラー: {pdf_path}\n→ {e}")

    def get_last_markdown(self) -> Optional[str]:
        """最後に処理したPDFのMarkdownを取得

        Returns:
            処理後のMarkdownテキスト、未処理の場合はNone
        """
        return self._last_markdown

    def _extract_with_azure_di(self, pdf_path: Path) -> str:
        """Azure Document Intelligence を使用してMarkdownを抽出"""
        logger.info(f"Extracting with Azure Document Intelligence: {pdf_path}")

        try:
            with open(pdf_path, "rb") as f:
                file_content = f.read()

            # Azure Document Intelligence で解析
            # extract_figures が有効な場合は図の画像データも取得
            analyze_kwargs = {
                "output_content_format": DocumentContentFormat.MARKDOWN,
            }
            if self.extract_figures:
                analyze_kwargs["output"] = ["figures"]

            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=file_content),
                **analyze_kwargs
            )

            result = poller.result()

            # 図情報を保存（後で処理するため）
            self._last_figures = result.figures if hasattr(result, 'figures') else None
            # operation_idを保存（図の画像取得に使用）
            if self.extract_figures and hasattr(poller, 'details') and poller.details:
                self._last_operation_id = poller.details.get('operation_id')
            else:
                self._last_operation_id = None

            if result.content:
                logger.info(f"Azure DI extracted {len(result.content)} characters")
                if self._last_figures:
                    logger.info(f"Azure DI found {len(self._last_figures)} figures")
                return result.content
            else:
                raise RuntimeError(
                    f"Azure DI がコンテンツを返しませんでした: {pdf_path}\n"
                    "→ PDFファイルが破損しているか、空の可能性があります"
                )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Azure DI でのPDF処理に失敗: {e}")

    def _process_azure_di_figures(self, markdown_text: str) -> str:
        """Azure DIの図をVision APIでテキスト化してMarkdownに統合"""
        if not self._last_figures or not self.vision_client:
            return markdown_text

        logger.info(f"Processing {len(self._last_figures)} figures with Vision API")

        # 図の説明を収集
        figure_descriptions = []
        for figure in self._last_figures:
            try:
                figure_id = figure.id if hasattr(figure, 'id') else None
                if not figure_id:
                    continue

                # キャプション取得
                caption = ""
                if hasattr(figure, 'caption') and figure.caption:
                    caption = figure.caption.content if hasattr(figure.caption, 'content') else str(figure.caption)

                # 図の画像データを取得
                try:
                    figure_response = self.di_client.get_analyze_result_figure(
                        model_id="prebuilt-layout",
                        result_id=self._last_operation_id,
                        figure_id=figure_id
                    )
                    # イテレータからバイトデータを読み取り
                    figure_data = b"".join(figure_response)
                    # Vision APIで分析
                    image_base64 = base64.b64encode(figure_data).decode('utf-8')
                    description = self._analyze_figure_with_vision(image_base64, caption)
                    logger.info(f"Figure {figure_id} analyzed successfully")
                except Exception as e:
                    logger.warning(f"Failed to get figure {figure_id}: {e}")
                    description = f"（図の取得に失敗）"

                figure_descriptions.append({
                    'id': figure_id,
                    'caption': caption,
                    'description': description
                })

            except Exception as e:
                logger.warning(f"Error processing figure: {e}")

        # <figure>タグを順番に置換
        # Azure DIは<figure>...</figure>形式で図を出力
        import re
        figure_pattern = r'<figure[^>]*>.*?</figure>'
        figure_matches = list(re.finditer(figure_pattern, markdown_text, re.DOTALL))

        logger.info(f"Found {len(figure_matches)} <figure> tags, have {len(figure_descriptions)} descriptions")

        # 後ろから置換して位置がずれないようにする
        for i, match in enumerate(reversed(figure_matches)):
            desc_idx = len(figure_matches) - 1 - i
            if desc_idx < len(figure_descriptions):
                desc = figure_descriptions[desc_idx]
                replacement = f"\n\n【図】{desc['caption']}\n{desc['description']}\n\n"
                markdown_text = markdown_text[:match.start()] + replacement + markdown_text[match.end():]
                logger.info(f"Replaced <figure> tag #{desc_idx + 1} with description")

        return markdown_text

    def _analyze_figure_with_vision(self, image_base64: str, caption: str = "") -> str:
        """Vision APIで図を分析してテキスト説明を生成"""
        try:
            prompt = f"""この図について詳細に説明してください。

キャプション: {caption if caption else "なし"}

以下の情報を含めてください:
1. 図の種類（グラフ、チャート、図表、写真など）
2. 主要な要素や構成
3. 読み取れるデータや数値
4. 図が示す傾向や重要なポイント

簡潔かつ正確に説明してください。"""

            response = self.vision_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return f"（図の分析に失敗しました）"

    def _convert_html_tables_to_text(self, markdown_text: str) -> str:
        """HTML形式のテーブルをLLMで自然なテキストに変換"""
        if not self.vision_client:
            return markdown_text

        # HTML表を検出
        html_table_pattern = r'<table[^>]*>.*?</table>'
        matches = list(re.finditer(html_table_pattern, markdown_text, re.DOTALL | re.IGNORECASE))

        if not matches:
            return markdown_text

        logger.info(f"Converting {len(matches)} HTML tables to text")

        # 後ろから処理（インデックスずれ防止）
        for match in reversed(matches):
            html_table = match.group()
            try:
                text_description = self._convert_single_html_table(html_table)
                start, end = match.span()
                markdown_text = markdown_text[:start] + f"\n\n{text_description}\n\n" + markdown_text[end:]
                logger.info(f"Converted HTML table ({len(html_table)} chars)")
            except Exception as e:
                logger.warning(f"Failed to convert HTML table: {e}")

        return markdown_text

    def _convert_single_html_table(self, html_table: str) -> str:
        """単一のHTMLテーブルをテキストに変換"""
        prompt = f"""以下のHTML表を、自然な日本語の説明文に変換してください。

要件:
- 表の内容を正確に伝える
- 数値や項目名を省略しない
- 行や列の関係性を明確にする
- 箇条書きではなく、文章形式で記述する

HTML表:
{html_table}

説明文:"""

        response = self.vision_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )

        return response.choices[0].message.content

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

            prompt = get_user_prompt("image_description")

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
        """Markdownテキストを階層的に分割（章→節→項→目の順で必要な場合のみ細分化）"""

        # ヘッダーレベル定義（上位から順に）
        header_levels = [
            ("#", "H1"),    # 章
            ("##", "H2"),   # 節
            ("###", "H3"),  # 項
            ("####", "H4"), # 目
        ]

        # 階層的に分割
        chunks_text = [markdown_text]
        current_metadata = [{}]  # 各チャンクのメタデータを追跡

        for header_marker, header_name in header_levels:
            new_chunks = []
            new_metadata = []

            for chunk, meta in zip(chunks_text, current_metadata):
                if len(chunk) <= self.chunk_size:
                    # サイズ内なら分割不要
                    new_chunks.append(chunk)
                    new_metadata.append(meta)
                else:
                    # このレベルのヘッダーで分割
                    splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[(header_marker, header_name)],
                        strip_headers=False
                    )
                    split_docs = splitter.split_text(chunk)

                    if len(split_docs) <= 1:
                        # 分割できなかった場合はそのまま
                        new_chunks.append(chunk)
                        new_metadata.append(meta)
                    else:
                        for doc in split_docs:
                            new_chunks.append(doc.page_content)
                            # メタデータをマージ
                            merged_meta = {**meta, **doc.metadata}
                            new_metadata.append(merged_meta)

            chunks_text = new_chunks
            current_metadata = new_metadata

        # 最終手段: まだ大きいものはRecursiveCharacterTextSplitterで分割
        final_chunks = []
        final_metadata = []

        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for chunk, meta in zip(chunks_text, current_metadata):
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
                final_metadata.append(meta)
            else:
                # 段落単位で分割
                split_texts = size_splitter.split_text(chunk)
                for text in split_texts:
                    final_chunks.append(text)
                    final_metadata.append(meta.copy())

        # ChunkDataリストを作成
        chunks = []
        for i, (chunk_text, meta) in enumerate(zip(final_chunks, final_metadata)):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # セクションパスを構築
            headers = []
            for h in ["H1", "H2", "H3", "H4"]:
                if h in meta:
                    headers.append(meta[h])
            section_path = ' › '.join(headers)

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
        self.async_client = None
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')

        if endpoint and api_key and AzureOpenAI:
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-02-15-preview"
                )
                import httpx
                self.async_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-02-15-preview",
                    timeout=httpx.Timeout(60.0, connect=10.0)
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
        prompt = get_user_prompt("paraphrase", text=text)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
            return ""

    def generate_qa(self, context: str) -> Dict[str, str]:
        """QAペア生成"""
        if not self.client or not context.strip():
            return {}
        prompt = get_user_prompt("qa", context=context)
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
        prompt = get_user_prompt("summary", text=text, max_length=max_length)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return ""

    def generate_keywords(self, text: str) -> List[Dict[str, str]]:
        """キーワード抽出"""
        if not self.client or not text.strip():
            return []
        prompt = get_user_prompt("keywords", text=text)
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
        prompt = get_user_prompt("elaboration", text=text)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Elaboration generation failed: {e}")
            return ""

    def generate_translation(self, text: str, lang: str = "英語") -> str:
        """翻訳"""
        if not self.client or not text.strip():
            return ""
        prompt = get_user_prompt("translation", text=text, lang=lang)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Translation to {lang} failed: {e}")
            return ""

    def generate_specialized_qa(self, context: str, qa_type: str) -> Dict[str, str]:
        """特定観点のQA生成"""
        if not self.client or not context.strip():
            return {}
        prompt = get_specialized_qa_prompt(context=context, qa_type=qa_type)
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
        prompt = get_user_prompt("discussion", text=text, turns=turns)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Discussion generation failed: {e}")
            return ""

    # --- 非同期メソッド（並列処理用） ---

    async def _call_llm_async(self, prompt: str) -> str:
        """非同期LLM呼び出し"""
        if not self.async_client:
            return ""
        response = await self.async_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()

    async def generate_paraphrase_async(self, text: str) -> str:
        if not self.async_client or not text.strip():
            return ""
        prompt = get_user_prompt("paraphrase", text=text)
        try:
            return await self._call_llm_async(prompt)
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
            return ""

    async def generate_qa_async(self, context: str) -> Dict[str, str]:
        if not self.async_client or not context.strip():
            return {}
        prompt = get_user_prompt("qa", context=context)
        try:
            response_text = await self._call_llm_async(prompt)
            json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")
            return {}

    async def generate_summary_async(self, text: str, max_length: int = 150) -> str:
        if not self.async_client or not text.strip():
            return ""
        prompt = get_user_prompt("summary", text=text, max_length=max_length)
        try:
            return await self._call_llm_async(prompt)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return ""

    async def generate_discussion_async(self, text: str, turns: int = 3) -> str:
        if not self.async_client or not text.strip():
            return ""
        prompt = get_user_prompt("discussion", text=text, turns=turns)
        try:
            return await self._call_llm_async(prompt)
        except Exception as e:
            logger.warning(f"Discussion generation failed: {e}")
            return ""

    async def generate_translation_async(self, text: str, lang: str = "英語") -> str:
        if not self.async_client or not text.strip():
            return ""
        prompt = get_user_prompt("translation", text=text, lang=lang)
        try:
            return await self._call_llm_async(prompt)
        except Exception as e:
            logger.warning(f"Translation to {lang} failed: {e}")
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

    # config/.envから環境変数を読み込み（スクリプト位置基準の絶対パス）
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    load_dotenv(PROJECT_ROOT / 'config' / '.env')

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
