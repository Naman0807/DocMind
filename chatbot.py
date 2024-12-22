import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pptx import Presentation
import tempfile
import PyPDF2
from docx import Document
import os


class DocumentRAGChatbot:
    def __init__(self):
        # Initialize session state variables if they don't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "current_file" not in st.session_state:
            st.session_state.current_file = None

    def save_api_key(self, api_key):
        """Save and validate API key"""
        if api_key:
            genai.configure(api_key=api_key)
            st.success("API key saved successfully!")
            return True
        else:
            st.error("Please enter an API key")
            return False

    def process_document(self, file):
        """Process uploaded document"""
        try:
            file_ext = os.path.splitext(file.name)[1].lower()

            if file_ext == ".pdf":
                texts = self.extract_text_from_pdf(file)
            elif file_ext == ".docx":
                texts = self.extract_text_from_docx(file)
            elif file_ext == ".pptx":
                texts = self.extract_text_from_pptx(file)
            else:
                st.error("Unsupported file format")
                return False

            st.session_state.vector_store = self.create_vector_store(texts)
            st.session_state.current_file = file.name
            return True

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

    def extract_text_from_pdf(self, file):
        """Extract text from PDF files"""
        pdf_reader = PyPDF2.PdfReader(file)
        texts = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                texts.append(f"Page {page_num}: {text}")
        return texts

    def extract_text_from_docx(self, file):
        """Extract text from Word documents"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        doc = Document(tmp_file_path)
        texts = []
        for para_num, para in enumerate(doc.paragraphs, 1):
            if para.text.strip():
                texts.append(f"Paragraph {para_num}: {para.text}")

        os.unlink(tmp_file_path)
        return texts

    def extract_text_from_pptx(self, file):
        """Extract text from PowerPoint files"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        prs = Presentation(tmp_file_path)
        texts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)
            if slide_text:
                texts.append(f"Slide {slide_num}: {' '.join(slide_text)}")

        os.unlink(tmp_file_path)
        return texts

    def create_vector_store(self, texts):
        """Create vector store from text content"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.create_documents(texts)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=st.session_state.api_key
        )
        return Chroma.from_documents(chunks, embeddings)

    def query_document(self, query: str, model_name: str = "gemini-1.5-flash") -> str:
        """Query the document content using RAG"""
        results = st.session_state.vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in results])

        model = genai.GenerativeModel(model_name)
        prompt = f"""Instruction:
Thoroughly analyze the context provided below and deliver a clear, accurate, and concise response to the question. Your answer must be derived exclusively from the information within the context—no assumptions or external knowledge are allowed. If the context does not include the required information, explicitly state: "The context does not provide the necessary information."

Context:
{context}

Question:
{query}

    Answer:

"""

        response = model.generate_content(prompt)
        return response.text


def main():
    st.set_page_config(page_title="DocMind", layout="wide")

    # Initialize the chatbot
    chatbot = DocumentRAGChatbot()

    # Sidebar
    with st.sidebar:
        st.title("DocMind")

        # API Key input
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if chatbot.save_api_key(api_key):
            st.session_state.api_key = api_key

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "docx", "pptx"],
            help="Upload a PDF, Word, or PowerPoint file",
        )

        if uploaded_file and st.button("Process Document"):
            if not hasattr(st.session_state, "api_key"):
                st.error("Please save your API key first")
            else:
                with st.spinner("Processing document..."):
                    if chatbot.process_document(uploaded_file):
                        st.success(f"Document processed: {uploaded_file.name}")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Chat cleared")

    # Main chat area
    st.title("DocMind ChatBot")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your message"):
        if not st.session_state.vector_store:
            st.error("Please upload and process a document first")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and add assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.query_document(prompt)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.write(response)


if __name__ == "__main__":
    main()
