import os
from typing import Any, Dict, List
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test12345")

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

title = "ai-app"
mcp   = FastMCP(title)

@mcp.prompt()
def search_prompt(query: str) -> str:
    """Prompt template for web search."""
    return f"You are a search assistant. Please list top results for: {query}."

@mcp.prompt()
def concept_prompt(results: List[str]) -> str:
    """Prompt template for concept extraction."""
    joined = "\n".join(results)
    return f"Extract key concepts from these search results:\n{joined}"

@mcp.prompt()
def graph_prompt(query: str) -> str:
    """Prompt template for graph query."""
    return f"Given concepts, fetch related details for: {query}."

@mcp.tool()
def web_search(query: str) -> List[str]:
    """Use LLM to simulate web search results."""
    prompt = search_prompt(query)
    resp = llm.invoke(prompt)
    return [line.strip() for line in resp.content.splitlines() if line.strip()]

@mcp.tool()
def concept_extractor(results: List[str]) -> List[Dict[str, Any]]:
    """Use LLM to extract concepts from results."""
    prompt = concept_prompt(results)
    resp = llm.invoke(prompt)
    concepts = [line.strip() for line in resp.content.splitlines() if line.strip()]
    return [{"concept": c, "detail": c} for c in concepts]

@mcp.tool()
def update_graph(concepts: List[Dict[str, Any]]):
    """Persist concepts into Neo4j."""
    with neo4j_driver.session() as session:
        for c in concepts:
            session.run(
                "MERGE (C:Concept {name:$name}) SET C.detail=$detail",
                name=c["concept"], detail=c["detail"]
            )

@mcp.tool()
def query_graph(query: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch concept nodes from Neo4j."""
    with neo4j_driver.session() as session:
        result = session.run(
            "MATCH (C:Concept) RETURN C.name AS concept, C.detail AS detail LIMIT 5"
        )
        return [{"concept": r["concept"], "detail": r["detail"]} for r in result]

@mcp.tool()
def rerank(context: Dict[str, Any]) -> List[Any]:
    """Simple pass-through reranker."""
    return context.get("query_graph", [])

@mcp.tool()
def responder(context: Dict[str, Any]) -> str:
    """Summarize pipeline results."""
    ws = context.get("web_search", [])
    ce = context.get("concept_extractor", [])
    qg = context.get("query_graph", [])
    return f"Found {len(ws)} results, extracted {len(ce)} concepts, queried {len(qg)} graph nodes."

@mcp.tool()
def fullPipeline(
    userId: str,
    query: str,
    preferences: Dict[str, Any]
) -> Dict[str, Any]:
    # 1) search + extract + graph insert
    searchResults = web_search(query)
    concepts      = concept_extractor(searchResults)
    update_graph(concepts)

    # 2) query + rerank
    graphResults = query_graph(query, preferences)
    finalResults = rerank({"query_graph": graphResults})

    # 3) persist preferences as JSON string
    prefsJson = json.dumps(preferences)
    with neo4j_driver.session() as session:
        session.run(
            '''MERGE (u:User {id:$uid})
            SET u.preferencesJson = $prefsJson''',
            uid=userId,
            prefsJson=prefsJson
        )

    # 4) return full context and summary
    summary = responder({
        "web_search":        searchResults,
        "concept_extractor": concepts,
        "query_graph":       finalResults
    })

    return {
        "summary": summary,
        "searchResults": searchResults,
        "concepts": concepts,
        "graphResults": finalResults
    }

@mcp.resource("ping://{msg}")
def ping_resource(msg: str) -> str:
    return f"PONG: {msg}"

@mcp.resource("concept://{concept}")
def concept_resource(concept: str) -> str:
    with neo4j_driver.session() as session:
        rec = session.run(
            "MATCH (C:Concept {name:$name}) RETURN C.detail AS detail",
            name=concept
        ).single()
    return rec["detail"] if rec else f"No detail for {concept}"

@mcp.resource("search://{query}")
def search_resource(query: str) -> List[str]:
    return web_search(query)

if __name__ == "__main__":
    mcp.run()
