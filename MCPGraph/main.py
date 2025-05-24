import os
from typing import Any, Dict, List
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
import re

load_dotenv()

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test12345")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    api_key=OPENAI_API_KEY,
    temperature=0
)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

title = "ai-app"
mcp   = FastMCP(title)


@mcp.prompt()
def explainGraphPrompt(graphData: List[Dict[str, Any]]) -> str:
    graphJson = json.dumps(graphData, indent=2)
    return f"""
    You are an assistant that explains concept graphs using arrows and short comments, followed by a clear summary.

    Input (JSON):
    {graphJson}

    Instructions:
    1. Summarize important relationships using this format:
       Concept A → Concept B : Brief explanation
       (Use plain text arrows, no markdown or formatting)

    2. Include up to 10 such lines. Prioritize relevance and clarity.
    3. After the list, write 2-3 sentences summarizing what these relationships reveal about the user's concept graph.
    4. Do not repeat concepts unnecessarily. Avoid generic or vague statements.
    """

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
def userMemoryPrompt(userId: str, memory: List[Dict[str, str]], question: str) -> str:
    memory_json = json.dumps(memory, indent=2)
    return f"""
        You are answering a question based only on a user's memory graph.
        
        User ID: {userId}
        
        Question: {question}
        
        User's memory:
        {memory_json}
        
        Instructions:
        - Base every answer solely on the provided memory.
        - If the answer cannot be determined from memory, say so clearly.
        - Be concise and accurate.
        """

@mcp.tool()
def memory_qa(userId: str, question: str) -> str:
    """Ask questions about user`s memory"""
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (u:User {id:$uid})-[:REMEMBERS]->(c:Concept)
            RETURN c.name AS concept, c.detail AS detail
        """, uid=userId)
        memory = [{"concept": r["concept"], "detail": r["detail"]} for r in result]

    prompt = userMemoryPrompt(userId, memory, question)
    resp = llm.invoke(prompt)
    return resp.content.strip()

@mcp.tool()
def web_search(query: str) -> List[str]:
    """Use LLM to simulate web search results."""
    prompt = search_prompt(query)
    resp = llm.invoke(prompt)
    return [line.strip() for line in resp.content.splitlines() if line.strip()]

@mcp.tool()
def concept_extractor(results: List[str]) -> List[Dict[str, Any]]:
    """Use LLM to extract, normalize, and dedupe concepts."""
    prompt = concept_prompt(results)
    resp = llm.invoke(prompt)
    raw = [line.strip() for line in resp.content.splitlines() if line.strip()]
    seen = set()
    concepts = []
    for r in raw:
        n = normalize(r)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            concepts.append({"concept": n, "detail": n})
    return concepts

def normalize(name: str) -> str:
    cleaned = re.sub(r'^[\-\*\d\.\s]+', '', name)
    return re.sub(r'\s+', ' ', cleaned).strip()

@mcp.tool()
def explain_graph(userId: str) -> str:
    """Explain the graph for the given user."""
    with neo4j_driver.session() as session:
        result = session.run("""
        MATCH (u:User {id:$uid})-[*1..2]->(c:Concept)
        RETURN DISTINCT c.name AS concept, c.detail AS detail
        """, uid=userId)
        data = [{"concept": r["concept"], "detail": r["detail"]} for r in result]

    prompt = explainGraphPrompt(data)
    resp = llm.invoke(prompt)
    return resp.content.strip()

def autoLinkWithGDS():
    """Automatically link concepts in the graph using Neo4j GDS."""
    with neo4j_driver.session() as session:
        session.run("MATCH (:Concept)-[r:RELATED_TO]-() DELETE r")

        session.run("CALL gds.graph.drop('conceptGraph', false);")

        session.run("""
        CALL gds.graph.project(
          'conceptGraph',
          'Concept',
          { REMEMBERS: { type: 'REMEMBERS', orientation: 'UNDIRECTED' } }
        );
        """)

        session.run("""
        CALL gds.nodeSimilarity.write(
          'conceptGraph',
          {
            topK: 5,
            writeRelationshipType: 'RELATED_TO',
            writeProperty: 'score'
          }
        );
        """)

        session.run("CALL gds.graph.drop('conceptGraph', false);")

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
        autoLinkWithGDS()

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
