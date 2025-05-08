# Güncellenmiş versiyon: Tesseract yerine EasyOCR kullanan versiyon

import streamlit as st
import os
from PIL import Image
import numpy as np
import easyocr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from decouple import config

GOOGLE_KEY = config("GOOGLE_GEMINI_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_KEY)
history = StreamlitChatMessageHistory()

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question which might reference chat history, 
    formulate a standalone question. Do NOT answer it."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor. Use the following context to solve the problem step-by-step. "
               "Do not skip steps. Show calculations where necessary.\n\n{context}"),
    ("human", "{input}"),
])

def clear_history():
    if "langchain_messages" in st.session_state:
        del st.session_state["langchain_messages"]

def process_file(file):
    with st.spinner("📄 Processing file..."):
        file_path = os.path.join("./", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file_path).convert("RGB")
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(np.array(image), detail=0)
            text = "\n".join(result)
            docs = [Document(page_content=text)]
        elif ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
        else:
            st.error("❌ Unsupported file type.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': False}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()

        question_answer_chain = create_retrieval_chain(retriever, qa_prompt)
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        crc = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        st.session_state.crc = crc
        st.success("✅ Document processed successfully.")

st.title("🧠 Math Solver – Adım Adım Çözen Asistan")
uploaded_file = st.file_uploader("📁 PDF veya Görsel yükle (jpg, png, pdf, txt)", type=["pdf", "jpg", "jpeg", "png", "txt"])
submit = st.button("📄 Dosyayı Yükle", on_click=clear_history)

if uploaded_file and submit:
    process_file(uploaded_file)

question = st.chat_input("💬 Bir matematik sorusu sor (örn. problemi çöz...)")

if question:
    with st.chat_message("user"):
        st.markdown(question)

        if "crc" in st.session_state:
            response = st.session_state.crc.invoke(
                {"input": question},
                config={"configurable": {"session_id": "math-session"}}
            )

            answer_text = getattr(response, "content", str(response))
            
            with st.chat_message("assistant"):
                st.markdown(answer_text)
            if isinstance(response, dict) and "context" in response:
                with st.expander("📄 Kullanılan İçerik"):
                    for doc in response["context"]:
                        st.markdown(doc.page_content)
else:
    st.error("📌 Önce bir dosya yüklemelisiniz.")
