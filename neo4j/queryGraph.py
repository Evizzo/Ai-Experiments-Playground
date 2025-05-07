from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def getRelevantSentences(query, limit=5):
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Sentence)
            WHERE toLower(s.text) CONTAINS toLower($q)
            RETURN s.text AS text
            LIMIT $limit
        """, q=query, limit=limit)
        return [r["text"] for r in result]

if __name__ == "__main__":
    results = getRelevantSentences("adventure")
    for r in results:
        print("-", r)
