# 🌐 ChatGroq: A RAG-Based Streamlit Application
This repository contains a Streamlit web application that implements a Retrieval-Augmented Generation (RAG) pipeline using Groq LLM, Google Generative AI embeddings, and FAISS. The app allows users to load content from a URL, generate embeddings, and perform context-aware question answering.

![Image](https://github.com/user-attachments/assets/98d982a1-806d-42fd-8538-bc46dc89196c)



# 📖 What is Retrieval-Augmented Generation (RAG)?
Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. It uses a retriever to fetch relevant documents and augments the input for a language model to generate responses based on the retrieved context.

In this project, we:

1. Use FAISS to manage embeddings for document chunks.

2. Retrieve the most relevant context for a query.

3. Pass the context to Groq LLM for precise, context-driven answers.


# ⚙️ What is Groq?
Groq LLM is a powerful, large language model that specializes in advanced reasoning and natural language processing tasks. This application leverages the Groq API for accurate question-answering.



# ✍️ What is a Prompt Template?
A Prompt Template defines the structure and language of queries sent to the language model. It ensures the LLM receives context and questions in a consistent, optimized format for better results.


# 🔗 What is a Chain?
In LangChain, a chain connects various components like retrievers, prompt templates, and language models. This project creates:

1. A document chain for combining documents with a prompt. 

2. A retrieval chain that integrates a retriever and the document chain to handle context-aware queries.




# 🖥️ What Does the Application Do?

## 🚀 Features

1. Load Documents: Input a URL, and the app fetches and processes the content.

2. Embed Documents: Splits content into chunks and generates embeddings using Google Generative AI.

3. Query Answering: Ask questions about the content, and the app retrieves relevant context to provide precise answers.

4. Similarity Search: View the relevant chunks used to generate the response.

# 🛠️ How It Works


### 1. Session State Management

• Embeddings: Initializes and stores embeddings in the session.

• Vectors: Manages FAISS-based document embeddings dynamically.

### 2. Document Processing

• URL Loader: Fetches content from a provided URL.

• Text Splitter: Breaks content into manageable chunks for efficient embedding.

### 3. Question Answering

• Combines a retriever (FAISS) with Groq LLM via a retrieval chain.

• Provides context-aware answers to user queries.



# 🏗️ Tech Stack

1. Streamlit: Interactive front-end for user input and visualization.

2. Groq LLM: Core language model for question answering.

3. LangChain: Framework for chaining together document loaders, retrievers, and LLMs.

4. Google Generative AI: For embeddings and similarity search.

5. FAISS: Efficient management of vector embeddings for document retrieval.





# Demo 📽

Below is a demonstration of how the application works:

![Demo of the Application](https://github.com/Abdelrahman-Amen/RAG_Groq_Integration/blob/main/Demo.gif)
