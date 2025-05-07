from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def createVectorIndex():
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX sentence_embedding_index IF NOT EXISTS
            FOR (s:Sentence) ON (s.embedding)
            OPTIONS { indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }}
        """)
    print("Vector index created.")

if __name__ == "__main__":
    createVectorIndex()
