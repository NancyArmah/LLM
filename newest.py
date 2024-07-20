import os
import streamlit as st
import pandas as pd
import docx2txt
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embedding technique
from dotenv import load_dotenv
import time

# Custom Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# Streamlit app setup
st.set_page_config(page_title="ADVANCED ANALYTICS BOT", page_icon="image.jpeg", layout="wide")

# Custom CSS to center the title and file uploader text
st.markdown("""
<style>
    .centered-title {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stFileUploader label {
        display: none;
    }
    .stFileUploader div div {
        justify-content: center;
    }
    .stFileUploader div div div {
        display: none;
    }
    .stTextInput input {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stTextInput input:focus {
        border: 2px solid #007BFF !important;
        outline: none !important;
    }
    .stTextInput input::placeholder {
        color: gray !important;
    }
</style>
""", unsafe_allow_html=True)

# Display the logo at the top left corner
st.sidebar.image("image.jpeg", use_column_width=True)

# Streamlit app setup with centered title
st.markdown('<h1 class="centered-title">ADVANCED ANALYTICS BOT</h1>', unsafe_allow_html=True)
#st.subheader("Helluuuuuurrrrr!!!! Welcome to my Document Q&A Application! yaaaayyyyy!🎉")
st.write("This application allows you to ask questions based on the context of provided documents.")

# Initialize the Groq language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# Define multiple prompt templates
qa_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the following document in a concise and clear manner.
    <context>
    {context}
    <context>
    """
)

extract_prompt = ChatPromptTemplate.from_template(
    """
    Extract the following information from the document.
    Information needed: {input}
    <context>
    {context}
    <context>
    """
)

# Create chains for each task
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
summary_chain = create_stuff_documents_chain(llm, summary_prompt)
extract_chain = create_stuff_documents_chain(llm, extract_prompt)

# Define function to create vector store from predefined documents and website
def vector_embedding():
    questions_dir = "./Questions"
    if not os.path.exists(questions_dir):
        st.error(f"The Questions directory does not exist at the specified path: {questions_dir}")
        return
    
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector store, please wait..."):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader(questions_dir)
            st.session_state.docs = st.session_state.loader.load()
            
            # Load content from website
            url = "https://bluechiptech.biz/"
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                website_text = soup.get_text(separator="\n", strip=True)
                
                # Create document from website content
                website_doc = Document(page_content=website_text)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                website_docs = text_splitter.split_documents([website_doc])
            except Exception as e:
                st.error(f"An error occurred while fetching the URL content: {e}")
                website_docs = []
            
            # Combine documents from predefined PDFs and website
            combined_docs = st.session_state.docs + website_docs
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(combined_docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is Ready")

# Initialize the vector store for predefined documents and website content
vector_embedding()

# Define function to create vector store from uploaded files
def vector_embedding_from_uploaded_files(uploaded_files):
    uploaded_docs = []
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            text = docx2txt.process(file_path)
            temp_txt_path = file_path.replace(".docx", ".txt")
            with open(temp_txt_path, "w") as f:
                f.write(text)
            loader = TextLoader(temp_txt_path)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
            temp_txt_path = file_path.replace(".xlsx", ".txt")
            with open(temp_txt_path, "w") as f:
                f.write(text)
            loader = TextLoader(temp_txt_path)
        else:
            continue
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(final_documents, embeddings)
        uploaded_docs.append((final_documents, vector_store))

    if uploaded_docs:
        combined_docs = st.session_state.final_documents + [doc for docs, _ in uploaded_docs for doc in docs]
        combined_vector_store = FAISS.from_documents(combined_docs, st.session_state.embeddings)
        return combined_vector_store
    return None

# Streamlit UI for uploading files and asking questions
#st.header("Upload Files and Ask a Question")
#st.write("Upload your PDFs, DOCX, or XLSX files and enter your question based on the documents provided to get an accurate response.")
uploaded_files = st.file_uploader("", type=["pdf", "docx", "xlsx"], accept_multiple_files=True, label_visibility="hidden")

if uploaded_files:
    if st.button("Create Vector Store from Uploaded Files"):
        with st.spinner("Creating vector store from uploaded files, please wait..."):
            uploaded_vector_store = vector_embedding_from_uploaded_files(uploaded_files)
        
        if uploaded_vector_store:
            st.session_state.vectors = uploaded_vector_store
            st.success("Combined Vector Store DB is Ready")

# Streamlit UI for asking questions
#st.write("Enter your question based on the documents provided and get an accurate response.")
with st.form(key='question_form'):
    prompt1 = st.text_input("Enter your question or a topic to search for real-time news")
    submit_button = st.form_submit_button(label='Submit')

# Function to select the appropriate chain based on query
def get_chain_for_query(query):
    if "summarize" in query.lower():
        return summary_chain
    elif "extract" in query.lower():
        return extract_chain
    else:
        return qa_chain

# Handling question submission with multiple chains
if submit_button and prompt1:
    try:
        if uploaded_files:
            with st.spinner("Creating vector store from uploaded files, please wait..."):
                uploaded_vector_store = vector_embedding_from_uploaded_files(uploaded_files)
            
            if uploaded_vector_store:
                st.session_state.vectors = uploaded_vector_store
                st.success("Combined Vector Store DB is Ready")
        
        # Select the appropriate chain based on the query
        selected_chain = get_chain_for_query(prompt1)
        
        # Retrieve relevant documents
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, selected_chain)
        
        with st.spinner("Processing your question, please wait..."):
            response = retrieval_chain.run(prompt1)
        
        st.success("Response Generated")
        st.write("### Response")
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Adding custom CSS to style the question form
st.markdown("""
<style>
    .stTextInput input {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stTextInput input:focus {
        border: 2px solid #007BFF !important;
        outline: none !important;
    }
    .stTextInput input::placeholder {
        color: gray !important;
    }
</style>
""", unsafe_allow_html=True)
