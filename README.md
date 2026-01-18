# FAQ-Chatbot-Using-LangChain-and-LLM-APIs

---

## 1. Name and Purpose of the Chatbot

NestThermoBot is an FAQ-based chatbot that will assist users with commonly asked questions about the Google Nest Thermostat. The primary objective is to provide the user with quick and easy-to-understand responses related to the setup, feature options and troubleshooting utilizing official Google Nest help documentation. Unlike previous attempts at providing assistance through guessing or random responses, NestThermoBot provides fast, accurate and relevant information regarding the user's inquiry by referencing actual product documentation support directly from Google Nest official support pages and utilizing a large language model to create a clear and concise response based upon that reference material.

---

## 2. NLP/LLM Methods Utilized

The primary method utilized by NestThermoBot to aid in the user’s query is Retrieval-Augmented Generation (RAG). In this method, the chatbot first identifies the most relevant and pertinent information contained within a knowledge base (in this instance the official Nest Thermostat support documentation pages) and passes that relevant and pertinent information to a Large Language Model (LLM) to generate the final response. By utilizing RAG, the potential for Hallucination is minimized and the accuracy and reliability of the response is greatly enhanced since the response generated is based upon real product documentation. The system also utilizes Text Embeddings to convert individual blocks of support text into vectors allowing the system to search and identify similar information based upon meaning versus simply searching by keywords. LangChain is being utilized to load and organize the document, segment the documents into chunks, perform the retrievals, and construct prompts.

---

## 3. Data Set Details

### 3.1 Data Set Source/Link

The data set utilized for training the chatbot is comprised of official product support/FAQ documentation provided by Google Nest, specifically the Nest thermostat beginner guide and the thermostat help center.

---

### 3.2 Number of Records

The dataset for this project is made up of content pulled from 2 official Google Nest Thermostat Support Pages. Upon pre-processing and text chunking, this dataset produced 29 text chunks. Each of these chunks is a section of the original documentation and serves as one record stored in the vector database. These chunks are created using a constant chunk size with overlap so that the system can keep the contextual flow between sections of text intact. During the execution of the application, the number of chunks was verified and then outputted to the console as part of the data loading process.

---

### 3.3 Number of Features

Each record in the dataset contains the following three main components:

- **Text Content** – The text that was extracted from the official Nest Thermostat documentation. The text will serve as the factual reference point for answering user questions.
- **Metadata (Source URL)** – Identifies where the text was originally obtained from. This metadata provides transparency into the source of each response provided by the chatbot.
- **Embedding Vector** – An embedded numerical version of the text content utilizing OpenAI's text-embedding-3-small model. Each embedding has a fixed length of 1,536 dimensions, allowing the system to search for similar semantic meanings using the Chroma vector database.

These embedding vectors allow the chatbot to evaluate the user's question(s), versus the stored documentation based upon meaning as opposed to keyword matching, thus providing more accurate and contextually aware retrieval of applicable information.

---

### 3.4 Description of Feature Types

| Feature Name | Description                                                               | Data Type       |
|-------------|---------------------------------------------------------------------------|-----------------|
| content     | Text content from the support pages, broken down into chunks              | String          |
| source      | Source URL of the original webpage                                        | String          |
| embedding   | Vector representation of the text to support similarity search            | Numeric Array   |

---

### 3.5 Pre-Processing Steps

- Load the web pages (HTML) via a web document loader.
- Extract readable text content from the loaded HTML pages.
- Split the extracted text content into overlapping chunks to retain contextual relevance.
- Create vector representations of the extracted text content to support vector similarity searches and store the vector representations in a Chroma vector database.

---

## 4. Libraries, Tool Kits, and Frameworks

- **LangChain**: Manages the flow of the chatbot workflow (load → split → embed → retrieve → generate).
- **Open AI API**: Supplies the LLM for generating responses and supplies the embedding model for creating vector representations of the text to facilitate similarity searches.
- **ChromaDB (via Chroma in LangChain)**: Stores the created vector representations and facilitates vector similarity search retrievals.
- **Python Standard Library (os, sys, typing)**: Utilized to establish environment variables and support the overall program structure.

---

## 5. Application Design and Implementation

The chatbot utilizes the RAG pipeline:

- **Data Collection**: The program collects official support web pages.
- **Chunking**: The content collected is segmented into smaller, overlapping sections so that the retrieval process functions better when answering specific user queries.
- **Embedding + Storage**: Each chunk is embedded into a vector representation and stored locally in a Chroma Vector Database.
- **Retrieval**: When a user submits a query, the system retrieves the top matching chunks based upon their semantic similarity.
- **Response Generation**: The LLM receives the user’s query and the retrieved context and generates a helpful and informative response.
- **Grounding Rule**: The prompt instructs the chatbot to indicate its uncertainty if the response generated does not reside within the context retrieved thereby reducing the likelihood of hallucination.

---

## 6. Instructions for Running the Chatbot

### Step 1: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows PowerShell




