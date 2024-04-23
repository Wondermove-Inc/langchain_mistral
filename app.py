from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st

import os

# Ensure necessary directories exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# Initialization or default settings
st.session_state['template'] = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
Context: {context}
History: {history}
User: {question}
Chatbot:"""

st.session_state['prompt'] = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=st.session_state['template'],
)

st.session_state['memory'] = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question")

# Ollama Setup
st.session_state['vectorstore'] = Chroma(persist_directory='jj',
                                         embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                             model="mistral"))

st.session_state['llm'] = Ollama(base_url="http://localhost:11434",
                                 model="mistral",
                                 verbose=True,
                                 callback_manager=CallbackManager([]))

# UI title
st.title("PDF Chatbot")

# Handling PDF upload
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
if uploaded_file is not None:
    st.write("Processing the PDF, please wait...")
    file_path = f"files/{uploaded_file.name}.pdf"
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with st.spinner("Analyzing your document..."):
            # Process PDF file
            loader = PyPDFLoader(file_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            all_splits = text_splitter.split_documents(data)
            st.session_state['vectorstore'] = Chroma.from_documents(
                documents=all_splits, embedding=OllamaEmbeddings(model="mistral"))
            st.session_state['vectorstore'].persist()

        st.session_state.retriever = st.session_state['vectorstore'].as_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state['llm'], chain_type='stuff', retriever=st.session_state.retriever, verbose=True,
            chain_type_kwargs={"verbose": True,  "prompt": st.session_state['prompt'], "memory": st.session_state['memory']})

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about the document:")
    if user_input:
        with st.spinner("Fetching your answer..."):
            response = st.session_state.qa_chain.invoke(user_input)
        st.write(response['result'])
else:
    st.write("Please upload a PDF file.")