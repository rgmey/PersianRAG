# PersianRAG

A retrieval-augmented generation (RAG) system for question answering over Persian-language PDF documents, with a Streamlit chat interface. Built to explore the retrieval and text-processing challenges specific to Persian, a right-to-left, morphologically distinct language that most RAG tooling isn't built with in mind.

## What it does

1. **Upload one or more PDFs** through the Streamlit interface.
2. **Ingestion**: text is extracted (`PyPDF2`), normalized for Persian orthography (`hazm`), split into word-based chunks, embedded using a multilingual sentence-transformer model, and indexed in a local FAISS vector store.
3. **Chat**: each question is embedded and matched against the FAISS index (dense retrieval, top-k configurable via a slider in the UI); the retrieved chunks are passed to an LLM (via an OpenAI-compatible API) with a Persian-language RAG prompt instructing it to answer only from the retrieved context.
4. **Output**: answers are shown in the chat UI, and every prompt/response pair is also logged and rendered to a correctly shaped Persian PDF.

## Persian-specific challenges addressed

- **Text normalization**: uses `hazm` (a Persian NLP library) to normalize text and handle sentence tokenization before chunking, rather than treating Persian text like English.
- **Right-to-left PDF rendering**: generating readable Persian PDF output required explicit handling that most PDF libraries don't provide out of the box, using `arabic-reshaper` to reshape Persian letterforms for correct contextual joining, and `python-bidi` to apply the bidirectional algorithm so text renders right-to-left correctly, alongside a bundled Persian font (Vazirmatn).
- **Multilingual embeddings**: uses a multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) rather than an English-only embedding model, since standard English embedding models perform poorly on Persian text.

## Tech Stack

Python · Streamlit · FAISS (`faiss-cpu`) · sentence-transformers · hazm · PyPDF2 · reportlab · arabic-reshaper · python-bidi · OpenAI-compatible LLM API

## Getting Started

```bash
git clone https://github.com/rgmey/PersianRAG.git
cd PersianRAG
pip install -r requirements.txt

# Set environment variables (e.g. in a .env file):
# MODEL_NAME=<model name>
# BASE_URL=<OpenAI-compatible API base URL>
# API_KEY=<your API key>

cd src
streamlit run app.py
```

Upload a PDF in the sidebar, adjust the number of retrieved chunks (k) with the slider, and ask questions in the chat box.

## Project Structure Note

The `src/` folder contains two retrieval implementations: `RAG.py` (a custom FAISS pipeline, used by the live `app.py` Streamlit app) and `retrieve.py` / `generate.py` / `run.py` (an earlier, LangChain-based exploration with hardcoded sample Persian text, not currently wired into the app). If you're presenting this repo, it's worth either removing the unused exploration files or noting clearly which path is the working entry point, currently a reader has to inspect imports to figure out that `app.py` is the one that runs.

## Limitations / Future Work

- Retrieval is dense-only (FAISS vector search); no keyword/hybrid retrieval, unlike the author's RAG-service project
- No retrieval evaluation harness (Hit@k, MRR, etc.) currently implemented for this project
- Single-turn retrieval per question; no conversation-aware query rewriting for follow-up questions
- No automated tests

## License

[MIT / Apache 2.0 / etc.]
