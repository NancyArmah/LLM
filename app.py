import os 
import streamlit as st  #UI
from langchain_groq import ChatGroq    #to create chatbot
from langchain_text_splitters import RecursiveCharacterTextSplitter     #to convert documents to chunks
from langchain.chains.combine_documents import create_stuff_documents_chain #to get relevant doc qna to setup context
from langchain_core.prompts import ChatPromptTemplate #create custom prompt template
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #embed vector store DB
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embedding technique

from dotenv import load_dotenv

load_dotenv()

## load GROQ and GOOGLE AI embeddings

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


st.title("Korzzzz Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided  context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./Questions") #data ingestion
        st.session_state.docs=st.session_state.loader.load() #docs load
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)




prompt1=st.text_input("Enter Question from the Document")

if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time


if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])


    #with streamlit expander
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("..............")
            