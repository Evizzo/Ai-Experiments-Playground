# 🧠 AI Experiments—AI Playground

> A playground for building and testing ideas with AI
> Each project lives in its own folder with all details inside.

## 📂 Projects

---

### 📁 **Gemma3 270M** – Local AI Model Training & Chat (WIP)

🚀 A local AI training environment for Google's Gemma3 270M model, optimized for Apple Silicon MacBooks. Features interactive chat capabilities, LoRA fine-tuning scripts, and MLX framework integration for efficient training on M-series chips. Currently in development with setup automation and performance monitoring tools.

---

### 📁 **aiAgentSecurtyResearch** – AI-Powered Security Investigation Tool

🤖 An autonomous security investigation system powered by ChatGPT that:
- Uses function/tool calling, makes intelligent decisions about which tools to use
- Maintains a chain of thought during investigations
- Adapts its investigation strategy based on findings
- Provides detailed reasoning for each action taken

⚡️ Core AI Features:
- Dynamic function calling for intelligent data gathering
- Real-time thought process tracking and logging
- Autonomous investigation flow with up to 5 iterations
- Contextual decision making based on previous findings

---

### 📁 **FineTuningSft**

This script provides a command-line interface to automate the process of fine-tuning an OpenAI model. 
It handles uploading the training data, creating the fine-tuning job, and monitoring its progress until completion.
It also calculates how much fine-tuning will cost you, and validates fine-tuning data.

---

### 📁 **MCPGraph**

A LLM-powered backend for building and interpreting user-specific knowledge graphs.  
It uses **FastMCP** for dynamic tool orchestration and **Neo4j** for graph-based memory and reasoning.

This project enables natural language queries to be transformed into structured graph data using simulated web search,
concept extraction, reranking, and automatic linking via Neo4j GDS.

🔁 The pipeline includes:

1. Simulated search (via LLM)
2. Concept extraction + deduplication
3. Graph storage (`REMEMBERS`)
4. GDS similarity linking (`RELATED_TO`)
5. Graph querying + reranking
6. Summary response generation

🧩 Each step is a modular **MCP tool**, allowing composable reasoning and API access (e.g. `fullPipeline`, `explain_graph`, `search://`, `concept://`).

⚙️ Tech used: FastMCP, Neo4j (GDS), LangChain, OpenAI or Gemini, Python 3.10+

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
