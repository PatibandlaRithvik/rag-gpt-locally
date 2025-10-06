
Database Documentation Assistant
A secure, privacy-first Retrieval-Augmented Generation (RAG) application for question-answering over your internal database design PDFs. Upload a schema or ER diagram PDF, click "Process," and ask questions about tables, relationships, constraints, diagrams, and best practices. All processing is local—no cloud APIs required!

Features

Semantic search using sentence embeddings

Transparent source passages behind every answer


Powered by local Ollama LLM (llama3.2:1b)

Professional, easy-to-use Streamlit UI

LangChain: Orchestrates the RAG workflow

PyPDFLoader: Parses PDF documents

Text Splitter: Cuts text into context-preserving chunks

Embeddings (MiniLM): Converts chunks & queries to meaning-vectors

Chroma Vector Store: Stores and retrieves chunks by similarity

RetrievalQA Chain: Builds prompt and calls local LLM (Ollama)

Ollama (llama3.2:1b): Writes contextual answers entirely local

Quick Start
1. Clone and enter the repo
bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
2. (Optional) Create a Python virtual environment
bash
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
.venv\\Scripts\\activate            # Windows
3. Install Python dependencies
bash
pip install -r requirements.txt

(streamlit
langchain
langchain-community
chromadb
sentence-transformers
pypdf)

4. Install Ollama and pull the model
Download Ollama for your OS

Start Ollama (follow instructions on site)

Download the LLM

bash
ollama pull llama3.2:1b
5. Run the app
bash
streamlit run app.py
How It Works
PDF Loaded: Extracts each page as a document

Chunking: Splits pages to context-rich chunks

Embeddings: Transforms text chunks to vectors representing meaning

Vector Store: Stores chunks for efficient semantic search

Ask a Question: Your question is also turned into a vector

Retrieval: Finds top-k most similar chunks to your question

LLM Generation: Sends context + question to local LLM for the answer

Display: Streamlit shows answer plus supporting source passages

Requirements
Paste this as your requirements.txt file:

text
streamlit
langchain
langchain-community
chromadb
sentence-transformers
pypdf
You also need:

Ollama (external binary, install separately)

Customization
Chunk size and overlap: Change in RecursiveCharacterTextSplitter

Retrieval depth: Change "k": 4 in vectorstore.as_retriever()

Prompt: Edit the template string in database_prompt

Embedding model: Swap HuggingFace models in HuggingFaceEmbeddings

LLM: Use different Ollama models for bigger/smaller/faster answers

Security
No cloud/external LLM calls: All document content and answers stay on your system.

Local vectorstore: Chunks are saved in ./chroma_db (delete this folder anytime to reset index).

Troubleshooting
Ollama not detected: Make sure Ollama server is running. Restart or check install.

Model download issues: Try ollama pull llama3.2:1b manually.

PDF not loading: Try a text-based PDF (image scans may need OCR).

Chroma errors: Delete ./chroma_db/ and retry.

Resource usage: Use smaller models and chunk sizes for large docs.



Contact & Support
Author: Rithvik

Email: rithvik158@gmail.com

