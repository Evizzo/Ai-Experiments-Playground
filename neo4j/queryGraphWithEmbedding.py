from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
model = SentenceTransformer("all-MiniLM-L6-v2")

def findSimilarSentences(query, limit=5):
    queryEmbedding = model.encode(query).tolist()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('sentence_embedding_index', $limit, $embedding)
            YIELD node, score
            RETURN node.text AS text, score
        """, embedding=queryEmbedding, limit=limit)
        return [(r["text"], r["score"]) for r in result]

if __name__ == "__main__":
    print("\n🔎 Write here (type 'exit' to quit):\n")
    while True:
        userInput = input("Your query: ").strip()
        if userInput.lower() in ["exit", "quit"]:
            break
        results = findSimilarSentences(userInput)
        print("\n📌 Top Matches:")
        for i, (text, score) in enumerate(results, 1):
            print(f"{i}. ({round(score, 3)}): {text[:200]}...\n")
