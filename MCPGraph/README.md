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

to conenct go on http://127.0.0.1:6280 and set command to uv and arguments to run --with mcp mcp run main.py
