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

def embedAndStore(filePath, source="wiki", chapter="1", limit=500):
    with open(filePath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    lines = lines[:limit]

    with driver.session() as session:
        for i, line in enumerate(lines):
            embedding = model.encode(line).tolist()
            session.run("""
                CREATE (s:Sentence {
                    id: $id,
                    text: $text,
                    embedding: $embedding,
                    source: $source,
                    chapter: $chapter
                })
            """, id=i, text=line, embedding=embedding, source=source, chapter=chapter)

if __name__ == "__main__":
    embedAndStore("input/book.txt", source="wiki", chapter="1")
