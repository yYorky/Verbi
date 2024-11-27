import streamlit as st
import logging
import os
import tempfile
from voice_assistant.audio import record_audio
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
import htmlTemplates  # Importing the HTML templates and CSS

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Streamlit app
st.set_page_config(page_title="Verbi RAG Chatbot", layout="wide")

# Add a header with an image
st.markdown("<h1 style='text-align: center;'>Verbi RAG Chatbot</h1>", unsafe_allow_html=True)
st.image("Verbi\static\chatbot image.png", use_column_width=True)

# Inject the CSS from htmlTemplates.py
st.markdown(htmlTemplates.css, unsafe_allow_html=True)

# Ensure `chat_history` is part of session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embeddings_initialized" not in st.session_state:
    st.session_state.embeddings_initialized = False

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# Sidebar for PDF upload
st.sidebar.title("Upload PDF for RAG")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document for RAG", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

# Initialize embeddings and vector store
def initialize_embeddings():
    if not uploaded_file:
        st.error("No file uploaded. Please upload a PDF.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    texts = [doc.page_content for doc in final_documents]
    vectors = FAISS.from_texts(texts, embeddings)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        human_prefix="User",
        ai_prefix="Verbi",
    )
    retriever = vectors.as_retriever(search_kwargs={"k": 5})

    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    st.session_state.embeddings_initialized = True
    st.sidebar.success("Embeddings and vector store initialized.")

if uploaded_file and not st.session_state.embeddings_initialized:
    with st.sidebar:
        st.write("Processing the uploaded document...")
        initialize_embeddings()

# Function to render chat messages using HTML templates
def render_chat_messages():
    chat_container = '<div class="chat-container">'
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "user":
            chat_container += htmlTemplates.user_template.replace("{{MSG}}", message["content"])
        elif message["role"] == "assistant":
            chat_container += htmlTemplates.bot_template.replace("{{MSG}}", message["content"]).replace("{{THOUGHT_PROCESS}}", message.get("thought_process", ""))
    chat_container += '</div>'
    st.markdown(chat_container, unsafe_allow_html=True)

# Function to handle voice assistant interaction
def handle_voice_assistant():
    if not st.session_state.get("conversation_chain"):
        st.error("Conversation chain not initialized. Please upload a PDF first.")
        return

    st.info("Recording... Speak now.")
    recorded_file = record_audio(Config.INPUT_AUDIO)

    if not recorded_file:
        st.warning("No audio recorded. Try again.")
        return

    transcription_api_key = get_transcription_api_key()
    user_input = transcribe_audio(
        Config.TRANSCRIPTION_MODEL, transcription_api_key, recorded_file, Config.LOCAL_MODEL_PATH
    )

    if not user_input:
        st.warning("Unable to transcribe audio. Try again.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.info(f"You said: {user_input}")

    response_text = "No document uploaded for context. Please upload a PDF."
    if st.session_state.embeddings_initialized:
        conversation_context = "\n".join(
            [f"{message['role']}: {message['content']}" for message in st.session_state.chat_history]
        )
        response = st.session_state.conversation_chain.invoke({"question": conversation_context})
        response_text = response["answer"]
    
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    st.success(f"Assistant: {response_text}")

# Main UI
st.markdown("<div style='position: fixed; top: 10px; width: 100%; text-align: center;'>", unsafe_allow_html=True)
if st.button("Record and Ask"):
    handle_voice_assistant()
st.markdown("</div>", unsafe_allow_html=True)

# Render chat messages
render_chat_messages()
