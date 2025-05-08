# 🧠 AI-Experiments

> A playground for building and testing ideas with AI
> Each project lives in its own folder with all details inside.

## 📂 Projects

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
