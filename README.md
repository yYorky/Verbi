# VERBI RAG - Voice-Assisted RAG Chatbot for PDFs 🎙️📄

![](https://raw.githubusercontent.com/yYorky/Verbi-RAG/refs/heads/main/static/chatbot%20image.png)

<p align="center">

## Motivation ✨

**Verbi RAG** is a Streamlit-based application designed to provide a seamless **voice-assisted Retrieval-Augmented Generation (RAG)** chatbot experience on **PDF documents**. 

The main unique point of this open-source chatbot is that it achieves very low latency by leveraging Cartesia AI's text to speech voice API (https://www.cartesia.ai/) and Groq's fast inference on hosted opensource models (https://console.groq.com/login). You can expect immediate response while chatting with the Verbi RAG about your document.

With an intuitive user interface, you can upload a PDF document, engage in natural conversations, and receive answers that leverage the uploaded document's context—all enhanced with voice input and output capabilities.

This app builds on the modular architecture of the original Verbi repository (https://github.com/PromtEngineer/Verbi) and adds a streamlit UI for end users to experiment with conversational AI backed by state-of-the-art tools for PDF-based knowledge retrieval.

---

## Features 🧰

- **Voice-Assisted Conversations**: Record your voice, interact with the chatbot, and listen to responses in real-time.
- **RAG Pipeline**: Upload a PDF document and query its content dynamically using Retrieval-Augmented Generation.
- **Interactive UI**: Built with Streamlit, featuring easy-to-use elements like file upload, buttons, and chat history display.
- **Modular Design**: Combines SOTA models for transcription, response generation, and text-to-speech (TTS).
- **Embeddings and Vector Search**: Utilizes FAISS for efficient document retrieval.
- **Customizable Models**: Options for transcription, TTS, and LLMs from OpenAI, Groq, Deepgram, and local alternatives.

---

## User Interface 🖥️

![](https://raw.githubusercontent.com/yYorky/Verbi-RAG/refs/heads/main/static/Verbi%20RAG.JPG)

### Upload PDFs
- Add your PDF file via the **sidebar** to load content into the app for context-aware RAG.
- Embedding and document splitting occur in the background.

### Chat History
- View a dynamically updated history of your conversation in the main interface.
- Both your queries and Verbi's responses are displayed in a clean, intuitive layout.

### Voice Assistant
- Click the **"Click to talk"** button to record your query.
- Receive concise and accurate responses via text and synthesized speech.

### Engaging Design
- Includes a header and chatbot logo styled with CSS for a polished user experience.

---

## Project Structure 📂

```plaintext
voice_assistant/
├── voice_assistant/
│   ├── __init__.py
│   ├── audio.py
│   ├── api_key_manager.py
│   ├── config.py
│   ├── transcription.py
│   ├── response_generation.py
│   ├── text_to_speech.py
│   ├── utils.py
│   ├── local_tts_api.py
│   ├── local_tts_generation.py
├── static/
│   ├── chatbot_image.png
├── setup.py
├── app_v7.py  # Streamlit app main file
├── requirements.txt
└── README.md
```

---

## Setup Instructions 📋

### Prerequisites ✅

- Python 3.10 or higher
- Virtual environment (recommended)
- Streamlit installed (`pip install streamlit`)

---

### Step-by-Step Instructions 🔢

1. 📥 **Clone the repository**

```shell
git clone https://github.com/yYorky/Verbi
cd Verbi
```

2. 🐍 **Set up a virtual environment**

Using `venv`:
```shell
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. 📦 **Install dependencies**
```shell
pip install -r requirements.txt
```

4. 🛠️ **Set up environment variables**
   - Create a `.env` file in the root directory.
   - Add your API keys:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

5. 🏃 **Run the Streamlit app**
```shell
streamlit run app_v7.py
```

---

## Usage 🕹️

1. Open the Streamlit app in your browser.
2. Upload a PDF document via the sidebar.
3. Click the **"Click to talk"** button to ask questions about the document.
4. Review responses in the chat window or listen to them via synthesized voice.

---

## Model Options ⚙️

Refer to the [original Verbi repository](https://github.com/PromtEngineer/Verbi) for detailed descriptions of the transcription, response generation, and TTS model options.

---

## Contributing 🤝

We welcome contributions! Fork the repository, make changes, and submit a pull request.

---

## License 📜

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yYorky/Verbi/blob/main/LICENSE) file for details.

