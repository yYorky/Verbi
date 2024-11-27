import streamlit as st
import threading
import logging
import time
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

# Global variables
stop_event = threading.Event()
assistant_thread = None

# Global variables for the app's state
conversation_chain = None
embeddings_initialized = False
vectors = None
chat_history = [{"role": "system", "content": """You are a helpful Assistant called Verbi. 
You are friendly, concise, and conversational. Your answers should be clear, short, and to the point, avoiding unnecessary details. 
When responding, prioritize the context provided by the user. However, if the user asks questions unrelated to the context (e.g., personal questions, general knowledge, or casual topics), 
begin by saying, 'That's not related to the current topic, but I'm happy to chat about it!' and then provide a friendly, thoughtful response. 
Maintain a warm and engaging tone throughout the conversation and aim to make the interaction enjoyable."""}]


# Initialize Streamlit app
st.set_page_config(page_title="Voice RAG Chatbot", layout="centered")

# Display image at the top
st.image("https://dailyaaj.com.pk/uploads/c49565f91b1cdd2eb8a646b6d4aa9771.jpg", use_container_width=True)

# Sidebar for PDF upload
st.sidebar.title("Upload PDF for RAG")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document for RAG", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

# Initialize embeddings and vector store
def initialize_embeddings():
    global conversation_chain, embeddings_initialized, vectors, system_prompt  # Declare as global
    if not uploaded_file:
        st.error("No file uploaded. Please upload a PDF.")
        return

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Use the uploaded file as a file-like object
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    # Extract the texts from the documents
    texts = [doc.page_content for doc in final_documents]

    # Create a FAISS vector store directly using the embedding model
    vectors = FAISS.from_texts(texts, embeddings)

    # Initialize the conversational retrieval chain
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        human_prefix="User",
        ai_prefix="Verbi",
        
    )
    
    retriever = vectors.as_retriever(search_kwargs={"k": 5})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )


    embeddings_initialized = True
    st.sidebar.success("Embeddings and vector store initialized.")

if uploaded_file and not embeddings_initialized:
    with st.sidebar:
        st.write("Processing the uploaded document...")
        initialize_embeddings()

# Inside the voice assistant logic (handle_voice_assistant function)
def handle_voice_assistant():
    global conversation_chain, embeddings_initialized, chat_history  # Declare as global
    while not stop_event.is_set():
        try:
            # Ensure conversation_chain is initialized before proceeding
            if conversation_chain is None:
                logging.error("Conversation chain not initialized. Exiting thread.")
                st.error("Conversation chain not initialized. Please upload a PDF first.")
                return

            # Record audio
            logging.info("Recording... Speak now.")
            recorded_file = record_audio(Config.INPUT_AUDIO, stop_event=stop_event)

            # Check if recording was stopped
            if recorded_file == "stopped" or stop_event.is_set():
                logging.info("Recording stopped.")
                break

            # Transcribe audio
            logging.info("Starting transcription.")
            transcription_api_key = get_transcription_api_key()
            user_input = transcribe_audio(
                Config.TRANSCRIPTION_MODEL, transcription_api_key, recorded_file, Config.LOCAL_MODEL_PATH
            )

            if not user_input:
                logging.info("No transcription detected. Retrying...")
                continue

            # Add user input to chat history
            chat_history.append({"role": "user", "content": user_input})
            logging.info(f"You said: {user_input}")

            # Check for special "goodbye" command
            if "goodbye" in user_input.lower():
                chat_history.append({"role": "assistant", "content": "Goodbye! Have a great day!"})
                logging.info("Assistant: Goodbye! Have a great day!")
                # Stop the assistant
                stop_event.set()
                return

            # Generate response using the RAG pipeline
            if embeddings_initialized and conversation_chain:
                response = conversation_chain.invoke({"question": user_input})
                response_text = response["answer"]
            else:
                response_text = "No document uploaded for context. Please upload a PDF."

            # Ensure the response is concise and conversational
            response_text = response_text.split('.')[0]  # Take only the first sentence to make it concise

            # Append assistant's response to chat history
            chat_history.append({"role": "assistant", "content": response_text})
            logging.info(f"Assistant: {response_text}")

            # Convert response to speech
            output_file = "output.mp3"
            tts_api_key = get_tts_api_key()
            text_to_speech(Config.TTS_MODEL, tts_api_key, response_text, output_file, Config.LOCAL_MODEL_PATH)

            # Play audio response
            play_audio(output_file, stop_event=stop_event)

        except Exception as e:
            logging.error(f"Error occurred: {e}")
            stop_event.set()
            break

# Display chat history dynamically with st.empty()
chat_container = st.empty()

def display_chat_history():
    """Render the chat history dynamically from global variable."""
    with chat_container.container():
        for message in chat_history:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            elif message["role"] == "assistant":
                st.markdown(f"**Verbi**: {message['content']}")

# Call display_chat_history() to update the UI with the latest chat history
display_chat_history()

# Start/Stop assistant buttons
if st.button("Start Assistant"):
    if not stop_event.is_set():
        stop_event.clear()
        assistant_thread = threading.Thread(target=handle_voice_assistant, daemon=True)
        assistant_thread.start()
        st.info("Voice assistant is running. Speak into your microphone.")
    else:
        st.warning("Voice assistant is already running.")

if st.button("Stop Assistant"):
    if assistant_thread and assistant_thread.is_alive():
        stop_event.set()
        assistant_thread.join()
        st.success("Voice assistant has stopped.")
    else:
        st.warning("Voice assistant is not running.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit & Verbi AI 🤖")
