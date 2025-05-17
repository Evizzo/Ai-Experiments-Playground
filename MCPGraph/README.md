To run neo4j locally with docker run
```
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test12345 \
  neo4j:5
```

to run do python main.py

* **GET /mcp/tools**
  Returns all registered MCP tool names.
* **POST /mcp/tools/{tool\_name}**
  Invoke a specific tool by name, passing JSON body as its params.
* **POST /plan**
  Takes `{ "query": "..." }`, asks the LLM for a step-by-step plan.
* **POST /execute**
  Takes `{ "plan": [...] }`, runs each action via MCP and returns the context + final answer.
* **POST /api/query**
  Shortcut that does `/plan` then `/execute` in one call, returns plan, context, answer.
* **POST /api/feedback**
  Logs whatever JSON you send it into `feedback_logs.jsonl` and returns `{ status: "ok" }`.
