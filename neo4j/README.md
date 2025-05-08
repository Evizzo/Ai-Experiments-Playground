To run neo4j locally with docker run
```
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test12345 \
  neo4j:5
```

Run createVectorIndex.py 

Run generateEmbeddings.py

Use queryGraphWithEmbedding.py to test

===========================================

run createEmbeddingsGraphsMetadataRelationships.py

CREATE INDEX FOR (b:Book) ON (b.title);
CREATE INDEX FOR (c:Chapter) ON (c.number);
CREATE INDEX FOR (s:Section) ON (s.number);

You can run ragAnswerGenerator.py to run hybrid, graph and vecto search for list of question for comparison purposes

---

# Hybrid Neo4j & Embeddings RAG Demo

A proof-of-concept Python toolkit that combines **vector (embedding) search** and **graph relationships** 
in Neo4j to power a simple RAG (Retrieval-Augmented Generation) pipeline with Google’s Gemini model.

---

## 🚀 Features

- **Ingestion**: Load JSON “Doctor Who” sample data into Neo4j as `Book` → `Chapter` → `Section` → `Text` nodes  
- **Embeddings**: Compute text embeddings via Sentence-Transformers and store on `Text` nodes  
- **Indexes**: Create both range/lookups and a Neo4j **vector index** for efficient similarity queries  
- **Hybrid Search**:  
  1. **Vector lookup** finds top-k relevant snippets  
  2. **Graph expansion** pulls all sibling texts in the same Section/Chapter  
- **RAG Answering**: Assemble context and prompt Gemini (via LangChain GoogleGen AI) to answer user questions  

---

## 📋 Prerequisites

1. **Docker & Neo4j**
2. Python 3.8+  
3. A Google API Key with access to your Gemini model

---

## ⚙️ Configuration

Create a `.env` in project root:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_API_KEY=key
LLM_MODEL_NAME=gemini-2.0-flash
```

---

## 🗄️ Data Preparation & Ingestion

1. **Edit** `input/data.json` with your JSON array of `{ book_title, chapter, chapter_title, section, author, publication_year, text }`
2. **Run** the ingestion script which will:

   * Clear existing graph
   * Create `Book`, `Chapter`, `Section`, `Text` nodes with `MERGE`
   * Compute embeddings (`paraphrase-MiniLM-L6-v2`)
   * Create indexes (range + vector index)

   ```bash
   python3 createEmbedingsGraphsMetadataRelationships.py
   ```

---

## 🔍 Hybrid Search & RAG

Ask natural-language questions over your graph+embeddings:

```bash
python3 ragAnswerGenerator.py
```

* **Vector hits** are printed (`🔍 [vector hits]`)
* **Graph expansions** follow (`↳ [graph adds …]`)
* Final answer from Gemini via LangChain
