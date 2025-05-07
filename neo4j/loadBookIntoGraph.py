from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def loadBook(filePath, maxLines=500):
    with open(filePath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    with driver.session() as session:
        for i, line in enumerate(lines[:maxLines]):
            session.run(
                "CREATE (:Sentence {id: $id, text: $text})",
                id=i, text=line
            )
    print(f"Loaded {min(maxLines, len(lines))} sentences.")

if __name__ == "__main__":
    loadBook("neo4j/input/book.txt")
