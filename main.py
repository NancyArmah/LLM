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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

# Streamlit app setup with improved layout
st.set_page_config(page_title="ADVANCED ANALYTICS BOT", page_icon="image.jpeg", layout="wide")

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0em;
        color: #0492C2;
    }
    .file-upload-area {
        padding: 1px;
        text-align: center;
        margin-bottom: 0em;
    }
    .stTextInput input {
        font-size: 1.2em;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #0492C2;
    }
    .stButton button {
        background-color: #0492C2;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #037099;
    }
    .response-area {
        background-color: #f0f0f5;
        border-radius: 10px;
        padding: 0px;
        margin-top: 0em;
    }
    .highlight {
        background-color: yellow;
    }
    .chat-bubble {
        padding: 10px;
        margin: 10px;
        border-radius: 10px;
    }
    .user-bubble {
        background-color: #d1e7dd;
        text-align: left;
    }
    .bot-bubble {
        background-color: #f0f0f5;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Display logo and title
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("image.jpeg", width=150)
with col2:
    st.markdown('<h1 class="main-title">ADVANCED ANALYTICS BOT</h1>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; font-size: 1em; margin-bottom: 0em;">This bot analyzes documents and answers questions based on their content.</div>""", unsafe_allow_html=True)

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

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to add message to conversation history
def add_to_conversation_history(user_message, bot_response):
    st.session_state.conversation_history.append({"user": user_message, "bot": bot_response})

# Function to display conversation history
def display_conversation_history():
    for entry in st.session_state.conversation_history:
        st.markdown(f'<div class="chat-bubble user-bubble">{entry["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bot-bubble">{entry["bot"]}</div>', unsafe_allow_html=True)

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
            except Exception as e:
                st.error(f"An error occurred while fetching the URL content: {e}")
                website_doc = None
            
            # Combine documents from predefined PDFs and website
            combined_docs = st.session_state.docs
            if website_doc:
                combined_docs.append(website_doc)
            
            # Split the combined documents once
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = text_splitter.split_documents(combined_docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.toast("Vector Store DB is Ready", icon="✅")

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

# Improved file uploader UI
st.markdown('<div class="file-upload-area">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded files
if uploaded_files:
    if st.button("Process Uploaded Files"):
        with st.spinner("Creating vector store from uploaded files..."):
            uploaded_vector_store = vector_embedding_from_uploaded_files(uploaded_files)
        if uploaded_vector_store:
            st.session_state.vectors = uploaded_vector_store
            st.toast("Vector Store DB Updated Successfully!", icon="✅")

# Improved question input and submit button
with st.form(key='question_form'):
    prompt1 = st.text_input("Ask me anything about the documents or website content:", placeholder="e.g., Summarize the main points of the document")
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 2])
    with submit_col1:
        submit_button = st.form_submit_button(label='Submit Question')

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
        # Select the appropriate chain based on the query
        selected_chain = get_chain_for_query(prompt1)
        
        # Retrieve relevant documents
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, selected_chain)
        
        # Generate response for the query
        with st.spinner("Generating response, please wait..."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            response_time = time.process_time() - start

        st.markdown('<div class="response-area">', unsafe_allow_html=True)
        st.write(f"Response time: {response_time:.2f} seconds")  # Corrected line
        st.write(response['answer'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Update conversation history
        add_to_conversation_history(prompt1, response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Display conversation history
display_conversation_history()

# Add a footer
st.markdown("---")
st.markdown("Powered by BlueChip Technologies | [Visit our website](https://bluechiptech.biz/)")
