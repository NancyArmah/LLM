import os
import streamlit as st #UI
from langchain_groq import ChatGroq #to create chatbot
from langchain_text_splitters import RecursiveCharacterTextSplitter #to convert documents to chunks
from langchain.chains.combine_documents import create_stuff_documents_chain #to get relevant doc qna to setup context
from langchain_core.prompts import ChatPromptTemplate #create custom prompt template
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #embed vector store DB, performs semantic/similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embedding technique
#from langchain.globals import set_verbose, get_verbose
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app setup
st.title("Korzzzz Document Q&A 😎")
st.subheader("Helluuuuuurrrrr!!!! Welcome to my Document Q&A Application! yaaaayyyyy!🎉")
st.write("This application allows you to ask questions to get specific answers based on the context of provided documents, get a summary of requests or extract specific information. Enjoy!!!")

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

# Define function to create vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector store, please wait..."):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./Questions")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is Ready")

# Streamlit UI for creating vector store
st.header("Setup Vector Store")
st.write("Click the button below to create the vector store from the documents.")
if st.button("Create Vector Store"):
    vector_embedding()

# Streamlit UI for asking questions
st.header("Ask a Question")
st.write("Enter your question based on the documents provided and get an accurate response.")
with st.form(key='question_form'):
    prompt1 = st.text_input("Enter Question from the Document")
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
        
        with st.spinner("Generating response, please wait..."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            response_time = time.process_time() - start

        st.write("Response time: {:.2f} seconds".format(response_time))
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("..............")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Custom CSS for better UI
st.markdown("""
<style>
    .stTextInput input {
        background-color: #f0f0f5;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        color: black; /* Set the text color to black */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
