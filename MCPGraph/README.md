# 🧠 AI Concept Graph Server

A LLM-powered backend for building and interpreting user-specific knowledge graphs.  
It uses **FastMCP** for dynamic tool orchestration and **Neo4j** for graph-based memory and reasoning.

This project enables natural language queries to be transformed into structured graph data using simulated web search,
concept extraction, reranking, and automatic linking via Neo4j GDS.

---

## 🔧 Tech Stack

* LangChain + Gemini (via Google Generative AI)
* FastMCP for modular tool registration
* Neo4j as a persistent concept graph
* Python 3.10+
* Docker (for Neo4j)

---

## 🚀 Features

* Simulated web search via LLM
* Concept extraction and graph storage
* Graph querying and basic reranking
* Preferences stored per user
* Custom URI routes (e.g. `search://`, `concept://`)
* LLM-based graph interpretation: explains user memory graphs and concept relationships in natural language
* Automatic concept linking using Neo4j Graph Data Science (GDS) similarity algorithms
* LLM-based QA over user memory: `memory_qa(userId, question)`
* Answers are grounded in what's explicitly stored in the concept graph

---

### ⚙️ How GDS Is Used

The system uses Neo4j's Graph Data Science (GDS) library to create semantic links between user concepts.

- A graph is projected in-memory using `REMEMBERS` edges
- Neo4j GDS computes node similarity using `nodeSimilarity.write`
- Top-k similar concepts are connected with `RELATED_TO` edges
- These relationships are later used by the `explain_graph` tool to summarize the user’s memory graph

## ▶️ Run Locally

### 1. Start Neo4j with GDS:

```bash
sudo docker stop neo4j-gds
sudo docker rm neo4j-gds
sudo docker run --name neo4j-gds \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test12345 \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=eval \
  -e NEO4JLABS_PLUGINS='["graph-data-science"]' \
  neo4j:5.12.0-enterprise
```

### 2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
```

### 3. Start MCP server and enter venv:

```bash
source .venv/bin/activate
mcp dev main.py
```

Or via [MCP Inspector](http://127.0.0.1:6280):

* Command: `uv`
* Args: `run --with mcp mcp run main.py`
* Set timeout to `90000`

---

## 🧪 Test the Full Pipeline

Run ```python test_mcp_tools.py``` or manually:

Use **MCP Inspector → Tools → `fullPipeline`** to simulate realistic knowledge graph creation:

---

### 🔁 Step 1: Build the Graph with Diverse Inputs

#### Query 1: performance engineering

```json
{
  "userId": "stef",
  "query": "lightweight sports cars with high agility and rear-wheel drive",
  "preferences": {
    "efficiency": "medium",
    "depth": 1
  }
}
```

#### Query 2: model examples

```json
{
  "userId": "stef",
  "query": "best petrol track cars under 100k in 2025",
  "preferences": {
    "maxPrice": 100000,
    "depth": 2
  }
}
```

#### Query 3: industry context

```json
{
  "userId": "stef",
  "query": "trends in performance automotive design and handling in 2025",
  "preferences": {
    "depth": 2
  }
}
```

---

### Query 4: Ask Questions About a User's Memory

```json
{
  "userId": "stef",
  "question": "What vehicles does stef remember related to handling or agility?"
}

```

---

### 📖 Step 2: Interpret the Graph
ℹ️ Behind the scenes, concepts are linked using Neo4j GDS's `nodeSimilarity` algorithm. 
Each new concept update triggers a similarity computation over all stored concepts, creating `RELATED_TO` links used later for explanations.

Use the `explain_graph` tool:

```json
{
  "userId": "stef"
}
```

---

## 🌐 Sample Endpoints

* `GET /search://neo4j`
* `GET /concept://LLM`
* `POST /explain_graph`  

## 🔧 MCP Components

### Prompts
- `search_prompt(query: str)`
- `concept_prompt(results: List[str])`
- `rerank_prompt(results, preferences)`
- `explainGraphPrompt(graphData: List[Dict])`
- `userMemoryPrompt(userId: str, memory: List[Dict[str, str]], question: str)`

### Tools
- `web_search(query: str)`
- `concept_extractor(results: List[str])`
- `update_graph(userId: str, concepts: List[Dict[str, Any]])`
- `query_graph(query: str, preferences: Dict)`
- `rerank(context: Dict)`
- `responder(context: Dict)`
- `fullPipeline(userId: str, query: str, preferences: Dict)`
- `explain_graph(userId: str)`
- `memory_qa(userId: str, question: str)`

---