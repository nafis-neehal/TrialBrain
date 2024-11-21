from streamlit.components.v1 import html
import base64

from dotenv import load_dotenv
import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import langchain
from typing import List, Dict
import fitz  # PyMuPDF for better PDF processing
from pathlib import Path

# Setup caching
langchain.llm_cache = InMemoryCache()

load_dotenv()

# Set page configuration
# st.set_page_config(
#     page_title="Clinical Trial Protocol Assistant", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


def extract_pdf_with_metadata(file_path: str) -> List[Dict]:
    """Extract text and metadata from PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Extract headers/section titles (simplified version)
        headers = [line.strip() for line in text.split('\n')
                   if line.strip() and len(line.strip()) < 100
                   and line.strip().isupper()]
        current_section = headers[0] if headers else "Unknown Section"

        metadata = {
            "source": Path(file_path).name,
            "page": page_num + 1,
            "section": current_section,
            "total_pages": len(doc)
        }

        chunks.append({"content": text, "metadata": metadata})

    doc.close()
    return chunks


def init_vector_store():
    """Initialize the vector store with document embeddings and metadata."""
    # Check if database exists
    if os.path.exists("./chroma_db"):
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        return vector_store

    documents = []
    pdf_dir = '../data/files/'

    # Process each PDF file
    for pdf_file in Path(pdf_dir).glob("**/*.pdf"):
        chunks = extract_pdf_with_metadata(str(pdf_file))
        documents.extend(chunks)

    # Enhanced text splitter with better overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?"]  # Reduced separators
    )

    # Split documents while preserving metadata
    splits = []
    for doc in documents:
        chunks = text_splitter.create_documents(
            texts=[doc["content"]],
            metadatas=[doc["metadata"]]
        )
        splits.extend(chunks)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Configure ChromaDB settings
    chroma_settings = {
        "anonymized_telemetry": False,
        "is_persistent": True,
        "persist_directory": "./chroma_db"
    }

    # Create vector store with optimized settings
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:M": 8,
        }
    )

    # Persist the vector store
    vector_store.persist()

    return vector_store


# Enhanced prompt template for better retrieval
QA_PROMPT = PromptTemplate(
    template="""You are an AI assistant helping with clinical trial protocols. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer.

Context: {context}

Question: {question}

Please provide a detailed answer and cite the specific sources (document name, page number, and section) where you found the information.

Previous conversation history: {chat_history}

Answer: """,
    input_variables=["context", "question", "chat_history"]
)


def get_conversation_chain(vector_store):
    """Create an enhanced conversation chain with the vector store."""
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        max_tokens=300,
        request_timeout=30,
        cache=True  # Enable LLM caching
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        k=3,
        return_messages=True,
        output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="mmr",
                                            search_kwargs={
                                                "k": 3
                                            }
                                            ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_separator": "\n",
        },
        return_source_documents=True,
        output_key="answer"
    )

    return conversation_chain


def format_source_reference(doc) -> str:
    """Format source reference with metadata."""
    metadata = doc.metadata
    return f"""
    📄 Source: {metadata.get('source', 'Unknown')}
    📑 Page: {metadata.get('page', 'Unknown')}
    📌 Section: {metadata.get('section', 'Unknown')}
    
    Content:
    {doc.page_content}
    """


def handle_user_input(user_question):
    """Process user input and generate response with enhanced source tracking."""
    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })

    # Extract source references - only unique sources
    sources = []
    seen_sources = set()  # To track unique combinations of source, page, and section

    if "source_documents" in response:
        for doc in response["source_documents"]:
            if doc.metadata:
                source_key = (
                    doc.metadata.get('source', 'Unknown'),
                    doc.metadata.get('page', 'Unknown'),
                    doc.metadata.get('section', 'Unknown')
                )
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append({
                        "source": source_key[0],
                        "page": source_key[1],
                        "section": source_key[2]
                    })

    # Get the answer from the response
    answer = response["answer"]

    # Format source citations (only unique sources)
    if sources:
        source_citation = "\n\nSources:\n" + "\n".join([
            f"- {s['source']}, Page {s['page']}, Section: {s['section']}"
            for s in sorted(sources, key=lambda x: (x['source'], x['page']))
        ])
        complete_answer = f"{answer}\n{source_citation}"
    else:
        complete_answer = answer

    # Update chat history
    st.session_state.chat_history.append((user_question, complete_answer))

    return response


def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    return pdf_display


def main():
    st.set_page_config(
        page_title="Clinical Trial Protocol Assistant", layout="wide")

    chat_col, pdf_col = st.columns([0.6, 0.4])

    with chat_col:
        st.title("Clinical Trial Protocol Assistant")
        st.write(
            "Ask questions about the clinical trial protocol documents in the knowledge base.")

        # Initialize vector store
        if st.session_state.vector_store is None:
            with st.spinner("Initializing knowledge base..."):
                st.session_state.vector_store = init_vector_store()
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vector_store)

        # Input section at top
        st.markdown("---")
        user_question = st.text_input(
            "Ask a question about the clinical trial protocol:", key="question_input")

        if st.button("Send", key="send_button"):
            if user_question:
                with st.spinner("Searching for answer..."):
                    response = handle_user_input(user_question)

                    if "source_documents" in response:
                        with st.expander("View Source Documents"):
                            seen_docs = set()
                            for doc in response["source_documents"]:
                                doc_key = (doc.metadata.get('source', 'Unknown'),
                                           doc.metadata.get('page', 'Unknown'),
                                           doc.metadata.get(
                                               'section', 'Unknown'),
                                           doc.page_content)
                                if doc_key not in seen_docs:
                                    seen_docs.add(doc_key)
                                    st.markdown(format_source_reference(doc))
                                    st.markdown("---")
                    st.rerun()

        # Display chat history in reverse order
        for question, answer in reversed(st.session_state.chat_history):
            st.markdown("---")
            st.markdown(f"👤 **Question**: {question}")
            if "\n\nSources:" in answer:
                main_answer, sources = answer.split("\n\nSources:", 1)
                st.markdown(f"🤖 **Answer**: {main_answer}")
                st.markdown(f"📚 **Sources**: {sources}")
            else:
                st.markdown(f"🤖 **Answer**: {answer}")

    with pdf_col:
        st.title("Protocol Document")
        pdf_path = "../data/files/TrialProtocolExample.pdf"
        pdf_html = display_pdf(pdf_path)
        html(pdf_html, height=800)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_TEAM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    main()
