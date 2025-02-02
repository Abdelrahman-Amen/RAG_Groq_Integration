import streamlit as st  # For building interactive web applications
import os  # To interact with the operating system and environment variables
import google.generativeai as genai  # To interact with Google Generative AI
from langchain_groq import ChatGroq  # For integrating the Groq LLM
from langchain_community.document_loaders import WebBaseLoader  # To load documents from a web URL
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For Google Generative AI embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into smaller chunks for processing
from langchain.chains.combine_documents import create_stuff_documents_chain  # To combine documents for processing
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain.chains import create_retrieval_chain  # To create a retrieval chain combining retrievers and LLMs
from langchain_community.vectorstores import FAISS  # For managing vector embeddings using FAISS
import time  # For tracking response time

from dotenv import load_dotenv  # To load environment variables from a .env file
load_dotenv()

# Load the Groq API key from environment variables
groq_api_key = os.environ['GROQ_API_KEY']

# Configure Google Generative AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure embeddings are initialized in session state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if the vector store exists in session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# Input field in the Streamlit app to accept a URL from the user
url_input = st.text_input("Enter a URL to load content from:")

# If the user provides a URL, process the content
if url_input:
    # Load documents from the provided URL
    st.session_state.loader = WebBaseLoader(url_input)
    st.session_state.docs = st.session_state.loader.load()

    # Split the loaded documents into manageable chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create vector embeddings for the document chunks
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    st.success("Documents loaded successfully from the provided URL!")

# Streamlit title for the app
st.title("ChatGroq Demo")

# Initialize the Groq LLM with the API key and model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define a prompt template for the question-answering task
prompt = ChatPromptTemplate.from_template("""
Using the context provided, respond accurately to the question below.
Make sure the response is based only on the context.
<context>
{context}
<context>
Question: {input}
""")

# Create a document chain for processing the input
document_chain = create_stuff_documents_chain(llm, prompt)

# Check if vector embeddings have been initialized
if st.session_state.vectors:
    # Create a retriever from the vector store
    retriever = st.session_state.vectors.as_retriever()
    
    # Create a retrieval chain combining the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Input field for the user to ask a question
    question = st.text_input("Enter your question:")

    if question:
        start = time.process_time()  # Start timer for response time tracking

        # Invoke the retrieval chain with the user's question
        response = retrieval_chain.invoke({"input": question})

        # Display the answer to the user's question
        st.write(response['answer'])

        # Expandable section to display the context of the relevant documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

        # Log and display the response time
        st.write("Response time:", time.process_time() - start)
else:
    st.warning("Please enter a URL to load documents first.")
