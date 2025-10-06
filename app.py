import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import subprocess

# -------- PAGE CONFIG & STYLE ---------
st.set_page_config(
    page_title="Database Documentation Assistant",
    page_icon="🗂️",
    layout="centered"
)

st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #f8fafc;
    }
    .main-title {
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -.8px;
        color: #20455e;
        padding-bottom: 0.2em;
    }
    .desc-section {
        color: #4a6572; font-size: 1.02rem; margin-bottom: 32px;
        max-width: 620px; margin-left:auto; margin-right:auto;
    }
    .section-title {
        font-size: 1.08rem; font-weight: 700;
        color: #114470; margin-top:2em; margin-bottom:0.6em;
    }
    .footer {
        text-align:center; margin-top:3.5em; font-size:.97rem; color: #8a99a8;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Database Documentation Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="desc-section">Upload your internal database design guide in PDF format. Ask questions about schema, relationships, ER diagrams, or design principles. All processing will stay within your secure environment—no cloud APIs or third-party sharing involved.</div>', unsafe_allow_html=True)

# -------- HELPER FUNCTIONS --------
def check_ollama_installed():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def install_ollama_model(model_name):
    try:
        with st.spinner("Preparing secure local AI environment..."):
            result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
            return result.returncode == 0
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

@st.cache_data
def load_and_process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        st.success(f"PDF loaded: {len(documents)} pages processed.")
    finally:
        os.unlink(tmp_file_path)
    return documents

@st.cache_resource
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=180)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore, len(texts)

DATABASE_PROMPT_TEMPLATE = """
You are a senior database designer. Use the provided context to answer detailed questions about tables, relationships, ER diagrams, and best practices.

Context: {context}

Question: {question}

Provide a concise, informative, and structured response.
"""
database_prompt = PromptTemplate(template=DATABASE_PROMPT_TEMPLATE, input_variables=["context", "question"])

def create_qa_chain(vectorstore):
    llm = Ollama(model="llama3.2:1b", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": database_prompt},
        return_source_documents=True
    )
    return qa_chain

# ----- SIDEBAR: CONFIG -----
with st.sidebar:
    st.markdown('<div class="section-title">Environment Check</div>', unsafe_allow_html=True)
    if not check_ollama_installed():
        st.error(
            "Ollama backend not detected. " 
            "Please ensure Ollama is installed (https://ollama.ai)."
        )
    else:
        st.info("Ollama backend active.")
        if st.button("Install AI Model (llama3.2:1b)"):
            if install_ollama_model("llama3.2:1b"):
                st.success("Model installed.")
            else:
                st.error("Model installation failed.")

# ------ MAIN INTERFACE -------
if check_ollama_installed():
    st.markdown('<div class="section-title">Upload Database Design PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select PDF file", type=['pdf'])
    if uploaded_file:
        st.success(f"File loaded: {uploaded_file.name}")
        if st.button("Process Document"):
            with st.spinner("Processing PDF document..."):
                documents = load_and_process_pdf(uploaded_file)
                vectorstore, num_chunks = create_vector_store(documents)
                qa_chain = create_qa_chain(vectorstore)
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain
                st.session_state.processed = True
                st.session_state.num_chunks = num_chunks
                st.session_state.num_documents = len(documents)
            st.success("Document indexed and ready for queries.")

    if st.session_state.get("qa_chain"):
        st.markdown('<div class="section-title">Ask a Documentation Question</div>', unsafe_allow_html=True)
        question = st.text_input("Type your question (e.g., Describe the relationship between User and Account tables):")
        if st.button("Get Answer", use_container_width=True) and question.strip():
            with st.spinner("Generating answer..."):
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                sources = result["source_documents"]
                st.markdown("**Response:**")
                st.write(answer)
                if sources:
                    with st.expander("References (top 2)"):
                        for i, source in enumerate(sources[:2]):
                            st.markdown(f"Reference {i+1}:")
                            st.text(source.page_content[:250] + "...")
    else:
        st.info("Complete the environment setup and upload a PDF to get started.")

st.markdown('<div class="footer">Database Documentation Assistant &nbsp;|&nbsp; Secure Local Processing</div>', unsafe_allow_html=True)
