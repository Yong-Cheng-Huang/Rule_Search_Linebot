"""建立法規 PDF 的 Chroma 向量資料庫。"""

import logging
import os
import re
import shutil
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    """集中管理法規向量庫建置參數。"""

    DATASETS_DIR = Path("datasets")
    PDF_PATTERNS = ("*.pdf", "*.PDF")

    CHROMA_PERSIST_DIR = "./chroma_db_langchain"
    COLLECTION_NAME = "regulations"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"

    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 80


ARTICLE_PATTERN = re.compile(r"(第\s*[0-9一二三四五六七八九十百千]+\s*條)")
PAGE_NOISE_PATTERNS = [
    re.compile(r"^\s*第\s*\d+\s*頁\s*$", re.MULTILINE),
    re.compile(r"^\s*頁\s*\d+\s*$", re.MULTILINE),
]


def find_pdf_files() -> list[Path]:
    """掃描 datasets 目錄內所有 PDF。"""
    if not Config.DATASETS_DIR.exists():
        raise FileNotFoundError(f"找不到資料目錄: {Config.DATASETS_DIR}")

    pdf_files: list[Path] = []
    for pattern in Config.PDF_PATTERNS:
        pdf_files.extend(sorted(Config.DATASETS_DIR.glob(pattern)))

    unique_files = sorted(set(pdf_files))
    if not unique_files:
        raise FileNotFoundError(f"在 {Config.DATASETS_DIR} 中找不到 PDF 檔案")
    return unique_files


def normalize_text(text: str) -> str:
    """清理 PDF 抽出的文字，保留法條結構。"""
    cleaned = text.replace("\u3000", " ").replace("\xa0", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    for pattern in PAGE_NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """逐頁讀取 PDF 內容。"""
    try:
        PdfReader = import_module("pypdf").PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少 pypdf 套件，請先執行 `pip install -r requirements.txt`。"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            logging.warning("PDF %s 第 %s 頁抽不出文字，已略過", pdf_path.name, index)
            continue

        pages.append(
            {
                "page": index,
                "text": page_text,
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "regulation_name": pdf_path.stem,
            }
        )

    return pages


def split_into_articles(pages: list[dict]) -> list[Document]:
    """把法規頁面內容切成條文級 Document。"""
    if not pages:
        return []

    combined_text = "\n\n".join(page["text"] for page in pages)
    matches = list(ARTICLE_PATTERN.finditer(combined_text))

    if not matches:
        regulation_name = pages[0]["regulation_name"]
        logging.warning("%s 未匹配到條號，將整份文件視為單一文件", regulation_name)
        return [
            Document(
                page_content=f"法規名稱：{regulation_name}\n條文內容：{combined_text}",
                metadata={
                    "source": pages[0]["source"],
                    "filename": pages[0]["filename"],
                    "regulation_name": regulation_name,
                    "page": pages[0]["page"],
                    "doc_type": "law_full_text",
                },
            )
        ]

    documents: list[Document] = []
    regulation_name = pages[0]["regulation_name"]
    source = pages[0]["source"]
    filename = pages[0]["filename"]
    first_page = pages[0]["page"]

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(combined_text)
        article_heading = re.sub(r"\s+", " ", match.group(1)).strip()
        article_body = combined_text[start:end].strip()
        if not article_body:
            continue

        documents.append(
            Document(
                page_content=(
                    f"法規名稱：{regulation_name}\n"
                    f"條號：{article_heading}\n"
                    f"條文內容：{article_body}"
                ),
                metadata={
                    "source": source,
                    "filename": filename,
                    "regulation_name": regulation_name,
                    "article_no": article_heading,
                    "page": first_page,
                    "doc_type": "law_article",
                },
            )
        )

    return documents


def build_langchain_documents() -> list[Document]:
    """從所有 PDF 建立條文級文件。"""
    all_documents: list[Document] = []
    pdf_files = find_pdf_files()

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        article_docs = split_into_articles(pages)
        all_documents.extend(article_docs)
        logging.info("已解析 %s，共 %s 筆條文文件", pdf_path.name, len(article_docs))

    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """對過長條文做第二次切塊。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    return splitter.split_documents(documents)


def reset_persist_dir() -> None:
    """重建 Chroma 持久化目錄，避免舊資料混入。"""
    persist_path = Path(Config.CHROMA_PERSIST_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        logging.info("已清除既有向量資料庫目錄: %s", persist_path)


def build_vectorstore(split_docs: list[Document]) -> int:
    """建立並持久化 Chroma 向量資料庫。"""
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY,
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=Config.CHROMA_PERSIST_DIR,
        collection_name=Config.COLLECTION_NAME,
    )
    return vectorstore._collection.count()


def main() -> None:
    """執行法規向量庫建置流程。"""
    logging.info("=== 開始建置法規向量資料庫 ===")

    if not Config.OPENAI_API_KEY:
        logging.error("請設定 OPENAI_API_KEY 環境變數")
        return

    try:
        documents = build_langchain_documents()
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return
    except Exception as exc:
        logging.error("載入 PDF 時發生錯誤: %s", exc)
        return

    if not documents:
        logging.error("沒有可用的法規內容可建立向量資料庫")
        return

    split_docs = split_documents(documents)
    logging.info("共載入 %s 份法規條文文件", len(documents))
    logging.info("切分後共有 %s 個向量片段", len(split_docs))

    try:
        reset_persist_dir()
        vector_count = build_vectorstore(split_docs)
    except Exception as exc:
        logging.error("建立向量資料庫時發生錯誤: %s", exc)
        return

    logging.info("向量資料庫建置完成")
    logging.info("collection: %s", Config.COLLECTION_NAME)
    logging.info("persist_directory: %s", Config.CHROMA_PERSIST_DIR)
    logging.info("vector_count: %s", vector_count)


if __name__ == "__main__":
    main()