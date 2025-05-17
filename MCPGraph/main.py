import os, json, uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from neo4j import GraphDatabase
from typing import Any, Dict, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
NEO4J_URI   = os.getenv("NEO4J_URI",   "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USER",  "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD", "test12345")
PORT        = int(os.getenv("PORT", "8000"))

llm           = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)
neo4j_driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
mcp           = FastMCP("ai-app")

tool_names: List[str] = []

def registerTool(fn):
    mcp.tool()(fn)
    tool_names.append(fn.__name__)
    return fn

@registerTool
def web_search(query: str) -> List[str]:
    return [f"Result for {query} #{i}" for i in range(3)]

@registerTool
def concept_extractor(results: List[str]) -> List[Dict[str, Any]]:
    return [{"concept": r, "detail": r} for r in results]

@registerTool
def update_graph(concepts: List[Dict[str, Any]]):
    with neo4j_driver.session() as session:
        for c in concepts:
            session.run("MERGE (C:Concept {{name:$name}})", name=c["concept"])

@registerTool
def query_graph(query: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    with neo4j_driver.session() as session:
        return [r.data() for r in session.run(
            "MATCH (C:Concept) RETURN C.name AS concept LIMIT 5"
        )]

@registerTool
def rerank(context: Dict[str, Any]) -> List[Any]:
    return context.get("query_graph", [])

@registerTool
def responder(context: Dict[str, Any]) -> str:
    ws = context.get("web_search", [])
    ce = context.get("concept_extractor", [])
    qg = context.get("query_graph", [])
    return f"Found {len(ws)} results, extracted {len(ce)} concepts, queried {len(qg)} graph nodes."

app = FastAPI(title="ai-app with MCP")

tools_router = APIRouter(prefix="/mcp", tags=["MCP Tools"])

@tools_router.get("/tools")
def list_tools():
    return tool_names

@tools_router.post("/tools/{tool_name}")
async def call_tool(tool_name: str, params: Dict[str, Any]):
    if tool_name not in tool_names:
        raise HTTPException(404, detail=f"Tool '{tool_name}' not found")
    result = await mcp.call_tool(tool_name, params)
    return result

app.include_router(tools_router)

@app.post("/plan")
def plan(payload: Dict[str, Any]):
    q = payload["query"]
    system_msg = (
        "You are an intelligent MCP planner. Given a user query, generate a minimal JSON list of tool calls.\n"
        "Each step must be a dict: {\"action\": tool_name, \"params\": {...}}.\n"
        "- `web_search` returns a list of strings like [\"result #0\", \"result #1\"]\n"
        "- `concept_extractor` must receive that list via {\"results\": [...]}\n"
        "- Final step is always `responder`, using a full context of previous results.\n"
        "- Output only raw JSON array, no comments or extra text.\n"
    )
    user_prompt = f"The user query is: \"{q}\".\nReturn a list of steps to answer it."
    prompt = system_msg + "\n\n" + user_prompt
    resp = llm.invoke(prompt)

    raw = resp.content.strip()
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        plan_json = json.loads(raw[start:end])
        if not isinstance(plan_json, list):
            raise ValueError("Output is not a list of steps")
        return {"plan": plan_json}
    except Exception as e:
        raise HTTPException(500, f"Bad plan JSON: {e} — Raw: {raw[:200]}")

@app.post("/execute")
async def execute(payload: Dict[str, Any]):
    steps = payload.get("plan", [])
    ctx: Dict[str, Any] = {}
    for step in steps:
        name   = step.get("action")
        params = step.get("params", {})
        result = mcp.call_tool(name, params)
        if hasattr(result, "__await__"):
            result = await result
        ctx[name] = result
    return {"context": ctx, "answer": ctx.get("responder")}

@app.post("/api/query")
async def query(payload: Dict[str, Any]):
    q  = payload.get("query")
    p  = plan({"query": q})
    ex = await execute({"plan": p["plan"]})
    return {"query": q, **ex}

@app.post("/api/feedback")
def feedback(payload: Dict[str, Any]):
    with open("feedback_logs.jsonl", "a") as f:
        f.write(json.dumps(payload) + "\n")
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
