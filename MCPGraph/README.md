To run neo4j locally with docker run
```
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test12345 \
  neo4j:5
```

install uv
```commandline
curl -LsSf https://astral.sh/uv/install.sh | sh

exec $SHELL -l
```

to run mcp dev main.py 

to connect go on http://127.0.0.1:6280 and set command to uv and arguments to run --with mcp mcp run main.py

set in configuration request Timeout to 90000

to test full pipeline go to mcp inspector, tools, use full pipeline tool, set username and query, and set prefrences, for example

username: stef

query: best sport petrol cars 2025

preferences: {
  "maxPrice": 300000,
  "efficiency": "low"
}

and you should see the results