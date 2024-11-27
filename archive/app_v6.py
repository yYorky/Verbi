import streamlit as st
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

# Initialize Streamlit app
st.set_page_config(page_title="Voice RAG Chatbot", layout="centered")

# Define the system prompt
system_prompt = """You are a helpful Assistant called Verbi. 
You are friendly, concise, and conversational. Maintain a warm and engaging tone throughout the conversation and aim to make the interaction enjoyable."""

# Ensure `chat_history` is part of session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize as empty list

if "embeddings_initialized" not in st.session_state:
    st.session_state.embeddings_initialized = False

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

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
    if not uploaded_file:
        st.error("No file uploaded. Please upload a PDF.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    # Extract the texts from the documents
    texts = [doc.page_content for doc in final_documents]

    # Create a FAISS vector store
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

# Function to handle assistant logic with STT and TTS
def handle_voice_assistant():
    if not st.session_state.get("conversation_chain"):
        st.error("Conversation chain not initialized. Please upload a PDF first.")
        return

    # Record audio from user
    st.info("Recording... Speak now.")
    recorded_file = record_audio(Config.INPUT_AUDIO)

    if not recorded_file:
        st.warning("No audio recorded. Try again.")
        return

    # Transcribe audio to text
    transcription_api_key = get_transcription_api_key()
    user_input = transcribe_audio(
        Config.TRANSCRIPTION_MODEL, transcription_api_key, recorded_file, Config.LOCAL_MODEL_PATH
    )

    if not user_input:
        st.warning("Unable to transcribe audio. Try again.")
        return

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.info(f"You said: {user_input}")

    # Truncate chat history to fit within context length
    max_context_length = 2000  # Adjust based on the model's token limit
    system_prompt_length = len(system_prompt.split())
    truncated_chat_history = []
    total_length = system_prompt_length

    for message in reversed(st.session_state.chat_history):
        message_length = len(message["content"].split())
        if total_length + message_length > max_context_length:
            break
        truncated_chat_history.insert(0, message)  # Prepend to preserve order
        total_length += message_length

    # Generate response using RAG
    response_text = "No document uploaded for context. Please upload a PDF."
    if st.session_state.embeddings_initialized:
        # Combine system prompt with truncated conversation history
        conversation_context = f"{system_prompt}\n" + "\n".join(
            [f"{message['role']}: {message['content']}" for message in truncated_chat_history]
        )
        response = st.session_state.conversation_chain.invoke({"question": conversation_context})
        response_text = response["answer"]
        
        # Ensure the response is concise and conversational
        response_text = response_text.split('.')[0]  # Take only the first sentence to make it concise

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    st.success(f"Assistant: {response_text}")

    # Convert response to speech
    output_file = "output.mp3"
    tts_api_key = get_tts_api_key()
    text_to_speech(Config.TTS_MODEL, tts_api_key, response_text, output_file, Config.LOCAL_MODEL_PATH)

# Main UI
st.header("Chat History")

# Display chat history dynamically
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Verbi:** {message['content']}")

# Record Button
if st.button("Record and Ask"):
    handle_voice_assistant()
