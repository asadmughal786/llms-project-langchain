# Pinecone + LangChain + OpenAI RAG Assistant

This project implements an end‑to‑end **Retrieval‑Augmented Generation (RAG)** pipeline using:

* Pinecone Vector Database
* LangChain
* OpenAI Embeddings & LLM
* PDF document ingestion
* Chunking & indexing
* Query answering from knowledge base
* Text‑to‑Speech output

## 📁 Project Structure

```
project/
├─ Documents/            # Upload PDFs here
├─ speech_to_text.py     # Custom function for audio playback
├─ main.py               # Main script (your provided code)
├─ .env                  # API keys
└─ README.md
```

## 🚀 Features

* Reads and loads PDF documents
* Splits text into chunks
* Embeds chunks using OpenAI embeddings
* Stores vectors in Pinecone
* Retrieves relevant chunks via cosine similarity
* Uses LangChain QA chain for final answer
* Plays the answer as speech

## 🧠 RAG Flow Diagram

```
PDFs → Text Chunks → OpenAI Embeddings → Pinecone Index
                                             ↓
User Query → Embedding → Pinecone Search → LangChain QA → Answer (Speech)
```

## 🔧 Requirements

Install dependencies:

```bash
pip install langchain langchain-openai langchain-community pinecone-client python-dotenv pypdf
```

(Optional, if your TTS uses additional libs install them too.)

## 🔑 Environment Variables (.env)

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

## 🏃‍♂️ Run the Script

```
python main.py
```

### ✅ On first run

* Loads and chunks PDFs
* Creates Pinecone index
* Embeds and inserts data

### On subsequent runs

* Detects index exists and skips upload

## 💬 Usage

Run and ask questions:

```
Enter your query: What is this PDF about?
```

Press `Q` to quit.

## 📦 Batching

Documents processed in batches to avoid OpenAI rate limits.

## ❗ Notes

* Update **index_name** if you want a clean start
* Update chunk size for large documents
* Works with any PDF placed in `Documents/`

## 📌 Future Enhancements

* Streamlit UI
* Add sources in responses
* Store metadata in Pinecone
* Add caching for queries

## 🛠 Troubleshooting

| Issue                    | Fix                                  |
| ------------------------ | ------------------------------------ |
| Rate limit error         | Script retries with delay            |
| Index exists but no data | Delete index from Pinecone dashboard |
| No audio output          | Check `speech_to_text` function      |

---

### 👤 Author

Mohammad Asad

Happy building with **RAG + Pinecone + OpenAI**! 🧠⚡️
