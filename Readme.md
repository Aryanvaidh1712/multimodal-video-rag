# Multimodal Video RAG Pipeline 🎥🤖

An end-to-end, privacy-first Retrieval-Augmented Generation (RAG) AI pipeline designed to transform raw, unstructured video courses into a fully searchable AI knowledge base. 

## 🚀 System Architecture
This project processes local video files, extracts the audio, generates timestamped transcripts, creates vector embeddings, and feeds the semantic context into a Large Language Model (LLM) to answer user queries accurately. 

* **Data Ingestion:** Extracts audio using `FFmpeg` and generates JSON transcripts via `OpenAI Whisper`.
* **Chunking & Vectorization:** Intelligently chunks text metadata using `Pandas` and creates vector embeddings using `BGE-M3`.
* **High-Speed Retrieval:** Serializes high-dimensional data into `Joblib` pickles and utilizes `scikit-learn`'s cosine similarity for sub-second context fetching.
* **LLM Integration:** Uses dynamic prompt engineering to feed context into local (`Ollama` / Llama 3.2) or cloud-based (`Google Gemini`) LLMs.

## 🛠️ Tech Stack
* **Language:** Python
* **Audio Processing:** FFmpeg
* **Transcription:** OpenAI Whisper
* **Embeddings & LLM:** Ollama (BGE-M3, Llama 3.2, DeepSeek-r1)
* **Data Processing & Math:** Pandas, NumPy, scikit-learn, Joblib

## ⚙️ Prerequisites
Before running this project, ensure you have the following installed on your system:
1. [FFmpeg](https://ffmpeg.org/download.html) (Added to system PATH)
2. [Ollama](https://ollama.ai/) (Running locally with your preferred models pulled)

## 📂 Project Structure
1. `video_to_mp3.py` - Batch converts course videos to MP3 format.
2. `mp3_to_json.py` - Transcribes audio to timestamped JSON using Whisper.
3. `preprocess_json.py` - Generates vector embeddings and saves them as a Joblib dataframe.
4. `process_incoming.py` - Handles user queries, performs cosine similarity search, and triggers the LLM.

## 🚀 How to Run (Execution Steps)

### **1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/Aryanvaidh1712/multimodal-video-rag
cd multimodal-video-rag
pip install -r requirements.txt
```

### **2. Prepare your data:**

Create the necessary directory structure and add your course materials:
- Create a folder named videos in the root directory.
- Place your .mp4 video lectures inside the videos folder.
- Ensure you have empty audios and jsons folders created as well.

### **3. Customize the AI Prompt:**

Open `process_incoming.py` and modify the `prompt` variable on line 47 to give the AI a specific personality or course context (e.g., "You are an AI assistant for a...").

### **4. Run the pipeline sequentially:**
This is an ETL pipeline, the scripts must be executed in order.
- **Step 1: Extract Audio**
```bash
python video_to_mp3.py
```

- **Step 2: Transcribe to Text (Whisper)**
```bash
python mp3_to_json.py
```

- **Step 3: Generate Vector Embeddings**
```bash
python preprocess_json.py
```

- **Step 4: Start the AI Assistant!**
```bash
python process_incoming.py
```