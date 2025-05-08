# 🧠 AI-Experiments

> A playground for building and testing ideas with AI
> Each project lives in its own folder with all details inside.

## 📂 Projects

---

### 📁 **neo4j** – Hybrid Graph + Embedding RAG Demo

🚀 A proof-of-concept that blends **Neo4j graph relationships** and **vector search** to power a Retrieval-Augmented Generation (RAG) system using **Google Gemini**.
It ingests structured “Doctor Who” data into a `Book → Chapter → Section → Text` graph, builds both **vector** and **graph** indexes, then performs **hybrid search** to answer user questions contextually.

🧰 Tech used: Docker, Python, Neo4j, Sentence-Transformers, LangChain, Gemini
📎 Scripts:

* `createVectorIndex.py` – Creates a vector index
* `generateEmbeddings.py` – Embeds raw text
* `queryGraphWithEmbedding.py` – Query via embedding
* `createEmbedingsGraphsMetadataRelationships.py` – Full ingestion & indexing
* `hybridVsGraphVsVector.py` – Compare search modes
* `ragAnswerGenerator.py` – RAG via LangChain & Gemini

---

### 📁 **voice-io-lang-graph** – Multi-Agent Voice Chat with LangGraph, Gemini & ElevenLabs

🗣️ A voice-enabled chat experience where you interact with four distinct AI personas—**Rick**, **Morty**, **Jerry**, and **Golubiro the paranoid pigeon spy**. Powered by **Google Gemini** for responses and **ElevenLabs** for real-time TTS output.

🧠 Built with **LangGraph** to declaratively orchestrate conversation turns:

1. You speak (via microphone or text).
2. LangGraph routes your message to the most relevant character.
3. That character responds.
4. Optionally, a second persona reacts to the reply (e.g. Golubiro always comments after Rick).
5. Loop back to your next input.

🧰 Tech used: LangGraph, LangChain, Google Gemini (via `langchain_google_genai`), ElevenLabs TTS, `speech_recognition`, Python 3.10+

---
