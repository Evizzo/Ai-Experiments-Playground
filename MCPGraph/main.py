import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
NEO4J_URI       = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER      = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS      = os.getenv("NEO4J_PASSWORD", "test12345")

llm           = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)
neo4j_driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

title = "ai-app"
mcp   = FastMCP(title)

@mcp.tool()
def web_search(query: str) -> List[str]:
    return [f"Result for {query} #{i}" for i in range(3)]

@mcp.tool()
def concept_extractor(results: List[str]) -> List[Dict[str, Any]]:
    return [{"concept": r, "detail": r} for r in results]

@mcp.tool()
def update_graph(concepts: List[Dict[str, Any]]):
    with neo4j_driver.session() as session:
        for c in concepts:
            session.run(
                "MERGE (C:Concept {name:$name}) SET C.detail=$detail",
                name=c["concept"], detail=c["detail"],
            )

@mcp.tool()
def query_graph(query: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    with neo4j_driver.session() as session:
        result = session.run(
            "MATCH (C:Concept) RETURN C.name AS concept, C.detail AS detail LIMIT 5"
        )
        return [{"concept": r["concept"], "detail": r["detail"]} for r in result]

@mcp.tool()
def rerank(context: Dict[str, Any]) -> List[Any]:
    return context.get("query_graph", [])

@mcp.tool()
def responder(context: Dict[str, Any]) -> str:
    ws = context.get("web_search", [])
    ce = context.get("concept_extractor", [])
    qg = context.get("query_graph", [])
    return f"Found {len(ws)} results, extracted {len(ce)} concepts, queried {len(qg)} graph nodes."

@mcp.resource("ping://{msg}")
def ping_resource(msg: str) -> str:
    return f"PONG: {msg}"

@mcp.resource("concept://{concept}")
def concept_resource(concept: str) -> str:
    with neo4j_driver.session() as session:
        rec = session.run(
            "MATCH (C:Concept {name:$name}) RETURN C.detail AS detail", name=concept
        ).single()
    return rec["detail"] if rec else f"No detail for {concept}"

@mcp.resource("search://{query}")
def search_resource(query: str) -> List[str]:
    return web_search(query)

@mcp.prompt()
def search_prompt(query: str) -> str:
    return f"You are a search assistant. Please search the web for: {query}."

@mcp.prompt()
def graph_prompt(query: str) -> str:
    return (
        f"Given the extracted concepts: {{concept_extractor}}, "
        f"query the knowledge graph for details on: {query}."
    )

if __name__ == "__main__":
    mcp.run()
