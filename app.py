from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = '63abd310-d5e9-4659-8c3c-c448144460f7'
PINECONE_INDEX_NAME = 'langchaingpthlive'
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment='us-east1')
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

def generate_embeddings(data):
    embeddings_model = OllamaEmbeddings(base_url='http://localhost:11434', model="mistral")
    return embeddings_model.embed_documents(data)

if not os.path.exists('files'):
    os.mkdir('files')

if 'template' not in st.session_state:
    st.session_state.template = "You are a knowledgeable chatbot; please provide details about your question."

if 'prompt' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions from the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
    
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(model="mistral", verbose=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    if not os.path.isfile(file_path):

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
        all_splits = text_splitter.split_documents(data)
        embeddings = generate_embeddings(all_splits)
        ids = [f'doc-{i}' for i in range(len(all_splits))]
        vectors = list(zip(ids, embeddings))
        pinecone_index.upsert(vectors)

    st.session_state.retriever = pinecone_index
   

    if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

        
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        response = st.session_state.qa_chain(user_input)
        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

else:
    st.write("Please upload a PDF file.")