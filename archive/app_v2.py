# Working RAG chain but only user text input

import streamlit as st
import threading
import time
import logging
import os
import tempfile
from voice_assistant.audio import record_audio, play_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.text_to_speech import text_to_speech
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_tts_api_key

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
POLLING_DELAY = 0.5  # Delay for polling listening state (in seconds)

# Initialize Streamlit app
st.set_page_config(page_title="Verbi Chatbot with RAG", layout="centered")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": """You are a helpful Assistant called Verbi. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are short and concise and in conversation form."""}
    ]

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "embeddings_initialized" not in st.session_state:
    st.session_state.embeddings_initialized = False

# Initialize a thread-safe event for listening state
listening_event = threading.Event()

# Sidebar for PDF upload
st.sidebar.title("Upload PDF for RAG")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document for RAG", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

# Initialize embeddings and vector store
def initialize_embeddings():
    if not uploaded_file:
        st.error("No file uploaded. Please upload a PDF.")
        return

    # Initialize embeddings
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Use the uploaded file as a file-like object
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    # Extract the texts from the documents
    texts = [doc.page_content for doc in final_documents]

    # Create a FAISS vector store directly using the embedding model
    st.session_state.vectors = FAISS.from_texts(texts, st.session_state.embeddings)

    # Initialize the conversational retrieval chain
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly set the output key
    )
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})

    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Explicitly set the output key
    )

    st.session_state.embeddings_initialized = True
    st.sidebar.success("Embeddings and vector store initialized.")

if uploaded_file and not st.session_state.embeddings_initialized:
    with st.sidebar:
        st.write("Processing the uploaded document...")
        initialize_embeddings()
        st.success("Embeddings and vector store initialized.")

# Display chat history
chat_container = st.empty()
def display_chat_history():
    """Render the chat history dynamically."""
    with chat_container.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Verbi:** {msg['content']}")

display_chat_history()

# Process user input and use RAG
def process_input(user_input):
    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.embeddings_initialized:
            response = st.session_state.conversation_chain({"question": user_input})
            response_text = response["answer"]
        else:
            response_text = "No document uploaded for context. Please upload a PDF for better answers."

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Convert response to speech
        tts_api_key = get_tts_api_key()
        output_file = "output.mp3"
        text_to_speech(Config.TTS_MODEL, tts_api_key, response_text, output_file, Config.LOCAL_MODEL_PATH)
        play_audio(output_file)

        display_chat_history()
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Input text bar
text_input = st.text_input("Ask a question or type your message:")

if text_input:
    process_input(text_input)

# Start/Stop button to toggle listening
if st.button("Start/Stop Recording"):
    if listening_event.is_set():
        listening_event.clear()
        st.info("Voice assistant stopped listening.")
    else:
        listening_event.set()
        st.success("Voice assistant is listening...")

# Voice assistant logic
def voice_assistant():
    try:
        while True:
            if listening_event.is_set():  # Use thread-safe event
                # Record audio
                record_audio(Config.INPUT_AUDIO)
                # Transcribe the audio to text
                transcription_api_key = get_transcription_api_key()
                user_input = transcribe_audio(Config.TRANSCRIPTION_MODEL, transcription_api_key, Config.INPUT_AUDIO, Config.LOCAL_MODEL_PATH)
                if user_input:
                    process_input(user_input)
            else:
                time.sleep(POLLING_DELAY)  # Polling delay to avoid high CPU usage
    except Exception as e:
        logging.error(f"Error in voice assistant thread: {e}")

# Start the assistant in a background thread
if "assistant_thread" not in st.session_state or not st.session_state.assistant_thread.is_alive():
    st.session_state.assistant_thread = threading.Thread(target=voice_assistant, daemon=True)
    st.session_state.assistant_thread.start()

# Footer
st.markdown("---")
st.caption("Powered by Streamlit & Verbi AI 🤖")