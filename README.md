# 🧠 AI Experiments - AI Playgorund

> A playground for building and testing ideas with AI
> Each project lives in its own folder with all details inside.

## 📂 Projects

---

### 📁 **MCPGraph** – W.I.P. 

Playing with MCP, tool calling, neo4j, fine tuning, LLM evaluations...

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

🗣️ A voice-enabled terminal chatbot where you interact with four chaotic AI personas—**Rick**, **Morty**, **Jerry**, and **Golubiro**, a paranoid drone-pigeon spy. 

Uses **Google Gemini** for natural language and **ElevenLabs** for speech.

🧠 Orchestrated with **LangGraph** as a declarative state machine:

1. You talk (via microphone or type).
2. LangGraph routes your message to the most relevant character using LLM-based classification.
3. The character responds.
4. If the character was Rick or Morty, a follow-up response is triggered by Golubiro or Jerry.
5. The flow loops back to capture the next input.

🧰 Tech used: LangGraph, LangChain, Google Gemini (`langchain_google_genai`), ElevenLabs TTS, `speech_recognition`, Python 3.10+

---
