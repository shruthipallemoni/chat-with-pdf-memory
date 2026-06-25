<div align="center">

<img src="https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
<img src="https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge&logo=groq&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector_Store-7B2FBE?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

# 📄 Chat with PDF — Memory

> **A memory-aware conversational RAG chatbot. Upload PDFs, ask questions, and get context-grounded answers — with full multi-turn conversation memory.**

[Features](#-features) · [Architecture](#️-architecture) · [Getting Started](#-getting-started) · [Usage](#-usage) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📌 Overview

**Chat with PDF (Memory)** is a Retrieval-Augmented Generation (RAG) application that lets you have intelligent, memory-aware conversations with the content of your PDF documents.

Built entirely with **LangChain LCEL** (no legacy chains), it automatically rewrites follow-up questions using chat history before retrieving relevant chunks — so you can ask *"what did it say about that?"* and still get the right answer.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📂 Multi-PDF Upload | Process multiple PDFs in a single session |
| 🧠 Conversation Memory | Full chat history tracked per session via `RunnableWithMessageHistory` |
| 🔄 Query Contextualization | Follow-up questions auto-rewritten into standalone queries before retrieval |
| ⚡ Groq Inference | Ultra-fast LLM responses via `llama-3.3-70b-versatile` |
| 🔍 Semantic Search | ChromaDB vector store with HuggingFace embeddings |
| 🧩 Pure LCEL Pipeline | Fully composable `|` operator chains, zero legacy wrappers |
| 🌐 Proxy Support | Works behind authenticated college/office network proxies |
| 🗂 Multi-Session | Separate conversation histories per session ID |

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────┐
                        │            User Question             │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │       Contextualization Chain        │
                        │                                      │
                        │  ChatPromptTemplate (history+input)  │
                        │            │                         │
                        │          Groq LLM                    │
                        │            │                         │
                        │      StrOutputParser                 │
                        │                                      │
                        │  → standalone, self-contained query  │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │         ChromaDB Retriever           │
                        │                                      │
                        │  all-MiniLM-L6-v2 Embeddings         │
                        │  RecursiveCharacterTextSplitter       │
                        │  chunk_size=5000 | overlap=500        │
                        │                                      │
                        │  → top-k relevant document chunks    │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │            QA Chain                  │
                        │                                      │
                        │  ChatPromptTemplate (context+history)│
                        │            │                         │
                        │          Groq LLM                    │
                        │            │                         │
                        │      StrOutputParser                 │
                        │                                      │
                        │  → grounded, concise answer          │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │   RunnableWithMessageHistory         │
                        │   → auto-saves turn to session store │
                        └─────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| UI Framework | Streamlit | Chat interface + file upload |
| LLM | Groq `llama-3.3-70b-versatile` | Answer generation + contextualization |
| Embeddings | `all-MiniLM-L6-v2` | Semantic text encoding |
| Vector Store | ChromaDB | Local similarity search |
| Orchestration | LangChain LCEL | Pipeline composition |
| PDF Loader | PyPDFLoader | Document ingestion |
| Memory | `RunnableWithMessageHistory` | Per-session chat history |
| Environment | python-dotenv | Secrets management |

---

## 📁 Project Structure

```
chat-with-pdf-memory/
├── app.py                  # Main Streamlit application (LCEL pipeline)
├── test_proxy.py           # Proxy connectivity test script
├── .env                    # Secrets — never committed
├── .env.example            # Template for environment variables
├── .gitignore              # Excludes secrets, cache, temp files
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- [Groq API Key](https://console.groq.com) — free tier available
- [HuggingFace Token](https://huggingface.co/settings/tokens) — free

---

### 1. Clone the repository

```bash
git clone https://github.com/lostmoon1513/chat-with-pdf-memory.git
cd chat-with-pdf-memory
```

### 2. Create and activate environment

```bash
conda create -n rag_genai python=3.10
conda activate rag_genai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

```env
# .env
HF_TOKEN=your_huggingface_token
```

> **Note:** `PROXY_USER` and `PROXY_PASS` are **only needed if you are on a restricted network** (college/corporate) where `huggingface.co` is blocked. If you have normal internet access, skip those variables entirely — the app works without them.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Usage

1. **Enter your Groq API key** in the sidebar
2. **Choose a model** from the dropdown (`llama-3.3-70b-versatile` recommended)
3. **Set a Session ID** — each unique ID maintains its own conversation history
4. **Upload PDFs** — one or multiple files supported
5. **Ask questions** in the chat input

**Example conversation:**

```
You:        What is the main topic of this document?
Assistant:  The document covers ...

You:        Can you elaborate on the second point?
Assistant:  The second point refers to ... [correctly uses chat history]
```

---

## 🌐 Proxy Configuration *(Optional — restricted networks only)*

> **Skip this section entirely if you have normal internet access.** This is only needed on networks that block `huggingface.co` (e.g. college hostels, corporate firewalls).

If you see errors like `407 Proxy Authentication Required` or `Max retries exceeded`, follow these steps:

**Step 1** — Add your proxy credentials to `.env`:
```env
# Add these ONLY if you are behind an authenticated proxy
PROXY_USER=your_proxy_username
PROXY_PASS=your_proxy_password
```

**Step 2** — Test that the proxy is working:
```bash
python test_proxy.py
# Expected output: ✅ Connected! 200
```

**Step 3** — Run the app normally:
```bash
streamlit run app.py
```

The app automatically routes all outbound requests through the proxy when these variables are present in `.env`.

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-core
langchain-community
langchain-groq
langchain-chroma
langchain-text-splitters
langchain-huggingface
sentence-transformers
chromadb
pypdf
python-dotenv
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | ✅ Always | HuggingFace token for embedding model download |
| `PROXY_USER` | 🔒 Only on restricted networks | Proxy username — skip if you have open internet |
| `PROXY_PASS` | 🔒 Only on restricted networks | Proxy password — skip if you have open internet |

---

## 🐛 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'ModelProfile'` | `langchain-huggingface` too new for your `langchain-core` | `pip install -U langchain-core langchain-huggingface` |
| `407 Proxy Authentication Required` | Network blocks HuggingFace | Set `PROXY_USER` + `PROXY_PASS` in `.env` |
| `model_decommissioned` | Groq retired the model | Use `llama-3.3-70b-versatile` |
| App stops immediately on launch | Silent crash during embedding load | Run `python test_proxy.py` to isolate the error |
| `No sentence-transformers model found` | Proxy blocking model download | Run `test_proxy.py` first; check credentials |

---

## 🗺️ Roadmap

- [ ] Streaming responses with `st.write_stream`
- [ ] Persistent vector store across sessions
- [ ] Multi-user session management
- [ ] Support for `.docx` and `.txt` file formats
- [ ] Deployment to Hugging Face Spaces

---

## 🧑‍💻 Author

<div align="center">

**Shruthi**  
CS & Engineering · RGUKT  
AI Engineer · GenAI Builder

[![GitHub](https://img.shields.io/badge/GitHub-shruthipallemoni-181717?style=for-the-badge&logo=github)](https://github.com/shruthipallemoni)

</div>

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with 🖤 using LangChain LCEL · Groq · Streamlit</sub>
</div>
