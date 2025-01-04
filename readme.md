# DocsMind - Document Analysis & Chat

A powerful document-based chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your documents using Google's Generative AI. DocsMind helps you analyze and interact with your documents through natural language queries.

## Demo: https://docsmind.streamlit.app/

## Features

- **Multi-Format Document Support**:
  - PDF files
  - Word documents (DOCX)
  - PowerPoint presentations (PPTX)
- **Advanced Document Analysis**:
  - Document summarization
  - Key points extraction
  - Context-aware question answering
- **RAG-based Question Answering**:
  - Uses Google's Generative AI (Gemini 1.5)
  - Efficient text chunking and embedding
  - FAISS vector store for fast similarity search
- **Interactive Interface**:
  - Clean, modern Streamlit UI
  - Split view for analysis and chat
  - Real-time document processing
  - Chat history preservation

## Prerequisites

- Python 3.8+
- Google API key (Gemini API access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naman0807/DocsMind.git
cd DocsMind
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run chatbot.py
```

2. Enter your Google API key in the sidebar

3. Upload your document (PDF, DOCX, or PPTX)

4. Use the document analysis tools:
   - Generate document summaries
   - Extract key points
   - Ask questions about the document content

## How It Works

1. **Document Processing**:
   - Extracts text from uploaded documents using specialized libraries for each format
   - Implements robust error handling for various document types

2. **Text Processing**:
   - Uses RecursiveCharacterTextSplitter for intelligent text chunking
   - Configurable chunk size and overlap for optimal context preservation

3. **Vector Store Creation**:
   - Embeds text chunks using Google's Embedding model
   - Creates FAISS vector store for efficient similarity search

4. **Question Answering**:
   - Retrieves relevant context using similarity search
   - Generates accurate answers using Gemini 1.5 model
   - Maintains conversation history

## Technologies Used

- **Streamlit**: Web interface and session management
- **LangChain**: Document processing and RAG implementation
- **Google Generative AI**: Text generation (Gemini 1.5) and embeddings
- **FAISS**: Vector similarity search
- **Document Processing**:
  - PyPDF2: PDF processing
  - python-docx: Word document processing
  - python-pptx: PowerPoint presentation processing

## Contributing

Feel free to contribute to this project! Visit the [GitHub repository](https://github.com/Naman0807/DocsMind) to:
- Report issues
- Submit pull requests
- Suggest improvements

## Author

Made by [Naman0807](https://github.com/Naman0807)
