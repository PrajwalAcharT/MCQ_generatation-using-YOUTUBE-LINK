from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import os
import shutil
from constant import CHROMA_SETTINGS

def clear_chroma_db():
    db_directory = CHROMA_SETTINGS.persist_directory
    try:
        if os.path.exists(db_directory):
            shutil.rmtree(db_directory)
        os.makedirs(db_directory)
        print("Chroma database cleared successfully!")
    except Exception as e:
        print(f"Error clearing database: {e}")

def create_embeddings_from_text(text):
    # Clear previous database
    clear_chroma_db()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents([text])
    
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Persist embeddings in Chroma
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_SETTINGS.persist_directory, 
        settings=CHROMA_SETTINGS
    )
    
    return db

def load_pdf_and_create_embeddings(uploaded_pdf):
    # Clear previous database data
    clear_chroma_db()

    # Load the PDF document
    loader = PDFMinerLoader(uploaded_pdf)
    documents = loader.load()

    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and persist in db
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_SETTINGS.persist_directory, 
        settings=CHROMA_SETTINGS
    )

    return db