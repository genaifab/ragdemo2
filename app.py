import fitz
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

@st.cache_resource
def load_and_process_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

st.title("🔍 Chat with your PDF using Ollama")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        retriever = load_and_process_pdf(uploaded_file)
        llm = Ollama(model="llama3.2")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                response = qa_chain.run(query)
                st.markdown("### 📘 Answer")
                st.write(response)
