# 🧠 AI Concept Graph Server

LLM-powered backend that uses **FastMCP** for tool orchestration and **Neo4j** as a graph memory to store and query concepts.

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

---

## ▶️ Run Locally

### 1. Start Neo4j:

```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test12345 neo4j:5
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

In MCP Inspector → Tools → `fullPipeline`, try:

```
userId: "stef"
query: "best sport petrol cars 2025"
preferences: {
  "maxPrice": 300000,
  "efficiency": "low"
}
```

You'll see:

* Simulated search results
* Extracted concepts
* Graph insertions
* Queried nodes and summary

---

## 🌐 Sample Endpoints

* `GET /search://neo4j`
* `GET /concept://LLM`
* `GET /ping://hello` → `PONG: hello`

## 🔧 MCP Components

### Prompts
- `search_prompt(query: str)`
- `concept_prompt(results: List[str])`
- `graph_prompt(query: str)`
- `rerank_prompt(results, preferences)`

### Tools
- `web_search(query: str)`
- `concept_extractor(results: List[str])`
- `update_graph(concepts: List[Dict])`
- `query_graph(query: str, preferences: Dict)`
- `rerank(context: Dict)`
- `responder(context: Dict)`
- `fullPipeline(userId: str, query: str, preferences: Dict)`

### Resources (HTTP schemes)
- `GET /ping://{msg}`
- `GET /search://{query}`
- `GET /concept://{concept}`

---