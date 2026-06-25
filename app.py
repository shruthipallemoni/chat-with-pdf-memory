## RAG Q&A Conversation with PDF Including Chat History (LCEL Version)

# app.py
import os
from dotenv import load_dotenv
load_dotenv()

PROXY_URL = f"http://{os.getenv('PROXY_USER')}:{os.getenv('PROXY_PASS')}@hostelinternet.rgukt.ac.in:3128"
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["HTTP_PROXY"]  = PROXY_URL   # ← these lines were MISSING
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["http_proxy"]  = PROXY_URL
os.environ["https_proxy"] = PROXY_URL

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")
st.title("📄 Conversational RAG with PDF & Chat History")
st.write("Upload PDFs and have a memory-aware conversation with their content.")

# ── Sidebar: API key + session ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    session_id = st.text_input("Session ID", value="default_session")
    st.caption("Each session keeps its own conversation history.")

# ── Session state init ────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store: dict[str, ChatMessageHistory] = {} # type: ignore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embeddings model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(llm, retriever):
    """
    Pure LCEL pipeline:

    1. contextualize_chain  – rewrites the user question using chat history so
                              the retriever always gets a self-contained query.
    2. rag_chain            – retrieves relevant chunks, then generates an answer
                              that is grounded in those chunks and aware of history.
    3. Wrapped with RunnableWithMessageHistory for automatic history injection.
    """

    # ── 1. Contextualization prompt ───────────────────────────────────────────
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, "
                "rewrite the question as a fully self-contained query. "
                "Do NOT answer it — only rewrite if needed, otherwise return it as-is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Produces a standalone string question
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    # ── 2. QA prompt ──────────────────────────────────────────────────────────
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the retrieved context below to "
                "answer the question. If the answer is not in the context, say so. "
                "Keep your answer concise (3 sentences max).\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # ── 3. Full LCEL chain ────────────────────────────────────────────────────
    #
    #  input dict: { "input": str, "chat_history": [messages] }
    #
    #  Step A: run contextualize_chain to get a standalone question string,
    #          keep original keys alongside.
    #  Step B: retrieve docs using the standalone question.
    #  Step C: format docs into a context string.
    #  Step D: pass everything to the QA prompt → LLM → parser.

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=contextualize_chain          # rewrites input
        )
        | RunnablePassthrough.assign(
            context=itemgetter("standalone_question")        # use rewritten Q
                    | retriever
                    | RunnableLambda(format_docs)
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with automatic history management
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# ── Main UI ───────────────────────────────────────────────────────────────────
if not api_key:
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.2)  
embeddings = get_embeddings()

# ── PDF upload ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Processing PDFs…"):
        documents = []
        for uploaded_file in uploaded_files:
            tmp_path = f"./temp_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        st.session_state.chain = build_chain(llm, retriever)

    st.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s).")

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

# Render existing messages
history = get_session_history(session_id)
for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# New user input
user_input = st.chat_input("Ask something about your PDF…")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = st.session_state.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
        st.write(answer)

# ── Optional: history debug panel ────────────────────────────────────────────
with st.expander("🗂 Raw chat history (debug)"):
    for msg in get_session_history(session_id).messages:
        prefix = "🧑" if isinstance(msg, HumanMessage) else "🤖"
        st.write(f"{prefix} {msg.content}")