import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import SimpleMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

os.makedirs('files', exist_ok=True)

api_key = "63abd310-d5e9-4659-8c3c-c448144460f7"
pc = Pinecone(api_key=api_key)
index_name = "mistral-docs"

index = pc.Index(index_name)

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://ollama:11434", model="mistral", verbose=True)

embeddings = OllamaEmbeddings(model="mistral")

st.title("Interactive PDF QA Bot")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    
    indexed_texts = {str(i): embeddings.embed_query(split) for i, split in enumerate(all_splits)}

    for i, vector in indexed_texts.items():
        index.upsert(vectors=[(str(i), vector)])

    def custom_retriever(query):
        query_vector = embeddings.embed_text(query)
        response = index.query(query_vector, top_k=5)
        combined_text = " ".join([all_splits[int(match['id'])] for match in response['matches']])
        return combined_text

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA(
            model=st.session_state.llm,
            retriever=custom_retriever,
            memory=SimpleMemory(),
            verbose=True
        )

    user_input = st.text_input("Your question:")
    if user_input:
        response = st.session_state.qa_chain.answer(user_input)
        st.write(response)
else:
    st.write("Please upload a PDF file.")