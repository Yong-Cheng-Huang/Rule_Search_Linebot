"""整合 LangChain 與 LINE Bot 的法規問答系統。"""

import logging
import os
import time

from dotenv import load_dotenv
from flask import Flask, abort, request
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    """集中管理應用程式設定。"""

    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4o-mini"

    CHROMA_PERSIST_DIR = "./chroma_db_langchain"
    COLLECTION_NAME = "regulations"

    EMBEDDING_MODEL = "text-embedding-3-small"
    SEARCH_K = 5
    ARTICLES_TO_CITE = 3
    RELEVANCE_THRESHOLD = 0.5


logging.info("正在載入 OpenAI 嵌入模型: %s...", Config.EMBEDDING_MODEL)

if not Config.OPENAI_API_KEY:
    logging.warning("警告：OPENAI_API_KEY 環境變數未設定，某些功能可能無法使用")

if not Config.LINE_CHANNEL_ACCESS_TOKEN:
    logging.warning("警告：LINE_CHANNEL_ACCESS_TOKEN 環境變數未設定，LINE Bot 功能將無法使用")
    Config.LINE_CHANNEL_ACCESS_TOKEN = "test_token"

if not Config.LINE_CHANNEL_SECRET:
    logging.warning("警告：LINE_CHANNEL_SECRET 環境變數未設定，LINE Bot 功能將無法使用")
    Config.LINE_CHANNEL_SECRET = "test_secret"

embeddings = None
vectorstore = None
llm = None
qa_chain = None

if Config.OPENAI_API_KEY:
    try:
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY,
        )
        logging.info("OpenAI 嵌入模型載入成功。")

        logging.info("正在從 '%s' 載入向量資料庫...", Config.CHROMA_PERSIST_DIR)
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME,
        )
        logging.info("向量資料庫載入成功。")

        llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0,
            api_key=Config.OPENAI_API_KEY,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.SEARCH_K})

        prompt_template = f"""
你是一位法規檢索助理。請根據以下提供的「相關法規條文」回答使用者問題。

重要指示：
1. 僅根據提供的條文內容回答，不要自行捏造法規內容。
2. 若找到相關條文，優先指出法規名稱與條號。
3. 先用精簡文字說明重點，再補充條文如何對應問題。
4. 若使用者問題明顯與法規無關，直接簡短回覆目前無法提供相關法規依據，不要列出法規名稱、條號或詳細規範。
5. 若資訊不足，請明確說明「依目前檢索到的條文，無法完全確認」。
6. 回答請使用繁體中文，避免提供法律保證或個案法律意見。
7. 只有在問題與法規確實相關時，才列出最多 {Config.ARTICLES_TO_CITE} 筆參考條文。

相關法規條文:
{{context}}

問題: {{question}}

回答:
"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        logging.info("LangChain QA 系統初始化成功。")
    except Exception as exc:
        logging.error("初始化 LangChain 組件時發生錯誤: %s", exc)
        logging.warning("將以基本模式運行，某些功能可能無法使用")
else:
    logging.warning("未設定 OPENAI_API_KEY，將以基本模式運行")


app = Flask(__name__)

configuration = Configuration(access_token=Config.LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(Config.LINE_CHANNEL_SECRET)

processed_tokens = set()
last_request_time = {}
REQUEST_COOLDOWN = 2


def format_regulation_info(metadata, content, index=1):
    """把條文 metadata 與內容整理成簡短訊息。"""
    regulation_name = metadata.get("regulation_name", "未知法規")
    article_no = metadata.get("article_no", "未標示條號")
    source = metadata.get("filename", "未知來源")

    compact_content = " ".join(content.split())
    if len(compact_content) > 220:
        compact_content = compact_content[:220] + "..."

    return (
        f"參考條文 {index}: {regulation_name} {article_no}\n"
        f"來源檔案: {source}\n"
        f"內容摘要: {compact_content}"
    )


def get_relevant_documents(query):
    """先用相似度門檻過濾明顯無關的問題。"""
    try:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=Config.SEARCH_K,
        )
    except Exception as exc:
        logging.warning("相關度搜尋失敗，改用一般搜尋: %s", exc)
        return vectorstore.similarity_search(query, k=Config.SEARCH_K)

    relevant_docs = []
    for doc, score in docs_with_scores:
        logging.info("檢索分數: %.4f, source=%s", score, doc.metadata.get("filename", "unknown"))
        if score >= Config.RELEVANCE_THRESHOLD:
            relevant_docs.append(doc)

    return relevant_docs


@app.route("/", methods=["GET"])
def index():
    """系統首頁路由。"""
    return """
    <h1>LangChain LINE Bot 法規問答系統</h1>
    <p>系統運行中！</p>
    <p>請透過 LINE Bot 進行法規查詢。</p>
    <hr>
    <p><strong>可用端點：</strong></p>
    <ul>
        <li><code>/</code> - 系統狀態頁面</li>
        <li><code>/callback</code> - LINE Bot Webhook</li>
        <li><code>/health</code> - 健康檢查</li>
    </ul>
    """


@app.route("/health", methods=["GET"])
def health_check():
    """系統健康檢查端點。"""
    try:
        if vectorstore:
            test_docs = vectorstore.similarity_search("法規", k=1)
            return {
                "status": "healthy",
                "message": "系統運行正常",
                "vectorstore_status": "connected",
                "documents_count": len(test_docs),
                "openai_status": "configured" if Config.OPENAI_API_KEY else "not_configured",
                "collection_name": Config.COLLECTION_NAME,
            }
        return {
            "status": "partial",
            "message": "系統運行中（基本模式）",
            "vectorstore_status": "not_available",
            "openai_status": "not_configured",
            "note": "請設定 OPENAI_API_KEY 並建立法規向量資料庫",
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "message": f"系統異常: {str(exc)}",
            "vectorstore_status": "error",
        }, 500


@app.route("/callback", methods=["POST"])
def callback():
    """LINE Bot webhook 回調端點。"""
    signature = request.headers.get("X-Line-Signature", "")
    if not signature:
        logging.error("缺少 X-Line-Signature 標頭")
        abort(400)

    body = request.get_data(as_text=True)
    logging.info("收到 webhook 請求，內容長度: %s", len(body))

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.error("簽章驗證失敗")
        abort(400)
    except Exception as exc:
        logging.error("Webhook 處理錯誤: %s", exc)
        abort(500)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """處理來自 LINE 使用者的文字訊息。"""
    logging.info("收到事件: %s", type(event).__name__)

    if not hasattr(event, "reply_token") or not event.reply_token:
        logging.warning("無效事件: 缺少 reply_token")
        return

    if not hasattr(event, "message") or not hasattr(event.message, "text"):
        logging.warning("無效事件: 非文字訊息")
        return

    if event.reply_token in processed_tokens:
        logging.warning("Reply token %s 已處理過，跳過", event.reply_token)
        return

    processed_tokens.add(event.reply_token)
    if len(processed_tokens) > 1000:
        processed_tokens.clear()

    user_query = event.message.text.strip()
    logging.info("接收到使用者查詢: %s (reply_token: %s)", user_query, event.reply_token)

    if not user_query or user_query.startswith("LineBot"):
        logging.info("忽略空訊息或系統訊息")
        return

    user_id = getattr(event.source, "user_id", "unknown")
    current_time = time.time()
    if user_id in last_request_time:
        time_since_last = current_time - last_request_time[user_id]
        if time_since_last < REQUEST_COOLDOWN:
            logging.info("請求太頻繁，忽略 (用戶: %s, 間隔: %.1f秒)", user_id, time_since_last)
            return

    last_request_time[user_id] = current_time

    try:
        if qa_chain and vectorstore:
            source_documents = get_relevant_documents(user_query)

            if not source_documents:
                llm_answer = "這個問題看起來與目前資料庫中的法規內容無直接關聯，因此無法提供相關法規依據。"
            else:
                result = qa_chain.invoke({"query": user_query})
                llm_answer = result.get("result", "抱歉，我無法處理您的請求。").strip()

            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=llm_answer)],
                )
            )

            if source_documents and user_id != "unknown":
                for idx, doc in enumerate(source_documents[: Config.ARTICLES_TO_CITE], start=1):
                    regulation_info = format_regulation_info(doc.metadata or {}, doc.page_content, idx)
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=regulation_info)],
                        )
                    )
        else:
            response_text = (
                f"您好！我收到了您的訊息：「{user_query}」\n\n"
                "目前系統正在基本模式運行中。\n"
                "要使用完整的法規問答功能，請管理員設定以下環境變數：\n"
                "• OPENAI_API_KEY\n"
                "• 重新建立向量資料庫\n\n"
                "感謝您的使用！"
            )
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=response_text)],
                )
            )
    except Exception as exc:
        logging.error("處理訊息時發生錯誤: %s", exc)
        try:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="系統發生一點問題，請稍後再試。")],
                )
            )
        except Exception as reply_exc:
            logging.error("發送錯誤訊息失敗: %s", reply_exc)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logging.info("啟動 Flask 應用程式於 port %s", port)
    app.run(host="127.0.0.1", port=port)
