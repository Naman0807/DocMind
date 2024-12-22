# Document RAG Chatbot

A powerful document-based chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your documents using Google's Generative AI.

## Features

- Support for multiple document formats:
  - PDF files
  - Word documents (DOCX)
  - PowerPoint presentations (PPTX)
- RAG-based question answering using Google's Generative AI
- Efficient text chunking and embedding
- Interactive Streamlit interface

## Prerequisites

- Python 3.8+
- Google API key

## Installation

1. Clone the repository

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## How It Works

1. **Document Processing**:
   The chatbot extracts text from uploaded documents (PDF, DOCX, or PPTX).

2. **Text Processing**:
   The extracted text is split into manageable chunks using RecursiveCharacterTextSplitter.

3. **Vector Store Creation**:
   Text chunks are embedded using Google's Embedding model and stored in a Chroma vector store.

4. **Question Answering**:
   User queries are processed using RAG technique:
   - Relevant context is retrieved from the vector store
   - Google's Generative AI model generates accurate answers based on the context

## Usage

1. Run the Streamlit app:

```bash
streamlit run chatbot.py
```

2. Upload your document

3. Ask questions about the document content

4. Receive accurate, context-based answers

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: Document processing and RAG implementation
- **Google Generative AI**: Text generation and embeddings
- **Chroma**: Vector store
- Various document processing libraries (PyPDF2, python-docx, python-pptx)
