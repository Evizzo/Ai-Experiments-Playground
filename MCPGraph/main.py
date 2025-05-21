import os
from typing import Any, Dict, List
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test12345")

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

title = "ai-app"
mcp   = FastMCP(title)

@mcp.prompt()
def rerank_prompt(results: List[Dict[str, Any]], preferences: Dict[str, Any]) -> str:
    """Prompt for ranking graph results based on user preferences."""
    results_json = json.dumps(results, indent=2)
    prefs_json = json.dumps(preferences, indent=2)
    return f"""
        You are an assistant that ranks graph results based on user preferences.
        User preferences: {prefs_json}
        Graph results: {results_json}
        Return a JSON array of objects, each having 'concept' and 'detail', ordered from most to least relevant.
        Example format:
        [
          {{ "concept": "X", "detail": "..." }},
          {{ "concept": "Y", "detail": "..." }}
        ]
        """

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

@mcp.prompt()
def should_link_prompt(conceptA: str, conceptB: str) -> str:
    return f"""
    Decide if there is a meaningful semantic relationship between these two concepts:
    - Concept A: {conceptA}
    - Concept B: {conceptB}
    
    Return only "yes" or "no" ONLY, and nothing else. If unsure or not applicable, return "no".
    """

def should_link(a: str, b: str) -> bool:
    prompt = should_link_prompt(a, b)
    resp = llm.invoke(prompt).content.strip().lower()
    return resp == "yes"

@mcp.tool()
def update_graph(userId: str, concepts: List[Dict[str, Any]]):
    """Persist concepts and user memory into Neo4j."""
    with neo4j_driver.session() as session:
        session.run("MERGE (u:User {id:$uid})", uid=userId)
        for c in concepts:
            session.run(
                """
                MERGE (c:Concept {name:$name})
                SET c.detail = $detail
                """,
                name=c["concept"], detail=c["detail"]
            )
            session.run(
                """
                MATCH (u:User {id:$uid}), (c:Concept {name:$name})
                MERGE (u)-[:REMEMBERS]->(c)
                """,
                uid=userId, name=c["concept"]
            )
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                a = concepts[i]["concept"]
                b = concepts[j]["concept"]
                if should_link(a, b):
                    session.run(
                        """
                        MATCH (a:Concept {name:$a}), (b:Concept {name:$b})
                        MERGE (a)-[:RELATED_TO]->(b)
                        """,
                        a=a, b=b
                    )

@mcp.tool()
def query_graph(userId: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch memory and related concepts from Neo4j."""
    depth = preferences.get("depth", 1)
    with neo4j_driver.session() as session:
        if depth > 1:
            cypher = """
            MATCH (u:User {id:$uid})-[:REMEMBERS]->(c:Concept)-[:RELATED_TO]->(rec:Concept)
            RETURN rec.name AS concept, rec.detail AS detail
            LIMIT 10
            """
        else:
            cypher = """
            MATCH (u:User {id:$uid})-[:REMEMBERS]->(c:Concept)
            RETURN c.name AS concept, c.detail AS detail
            LIMIT 5
            """
        result = session.run(cypher, uid=userId)
        return [{"concept": r["concept"], "detail": r["detail"]} for r in result]

@mcp.tool()
def rerank(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LLM-based reranker using the precise rerank_prompt."""
    results = context.get("query_graph", [])
    preferences = context.get("preferences", {})
    if not results or not preferences:
        return results

    prompt = rerank_prompt(results, preferences)
    resp = llm.invoke(prompt)
    try:
        ranked = json.loads(resp.content)
        if all(isinstance(item, dict) and 'concept' in item and 'detail' in item for item in ranked):
            return ranked
    except json.JSONDecodeError:
        pass
    return results

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
    # 1) search + extract
    searchResults = web_search(query)
    concepts = concept_extractor(searchResults)

    # 2) persist into graph-based memory
    update_graph(userId, concepts)

    # 3) query memory graph (with optional depth)
    graphResults = query_graph(userId, preferences)

    # 4) rerank & summarize
    finalResults = rerank({
        "query_graph": graphResults,
        "preferences": preferences
    })
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
    print("» Starting MCP server …")
    mcp.run()
    print("» MCP server started.")
