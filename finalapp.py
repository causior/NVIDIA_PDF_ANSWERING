import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
import tempfile  # For handling uploaded files

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = NVIDIAEmbeddings()
            
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name  # Get the temporary file path
            
            # Load the PDF using the temporary file path
            pdf_loader = PyPDFLoader(temp_file_path)
            st.session_state.docs = pdf_loader.load()  # Document Loading
            
            # Split documents into smaller chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
            
            # Create vector embeddings
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.write("Vector Store created successfully.")
        except Exception as e:
            st.error(f"Error creating vector embeddings: {str(e)}")

# Streamlit App Title
st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Text input for user questions
prompt1 = st.text_input("Enter Your Question From the Documents")

if uploaded_file and st.button("Documents Embedding"):
    vector_embedding(uploaded_file)

if prompt1 and "vectors" in st.session_state:  # Ensure vectors are initialized
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start} seconds")

        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):  # Ensure "context" exists
                st.write(doc.page_content)
                st.write("--------------------------------")
    except KeyError as ke:
        st.error(f"Missing key in response: {str(ke)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
