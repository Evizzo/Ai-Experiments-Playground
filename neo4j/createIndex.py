from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def createTextIndex():
    with driver.session() as session:
        session.run("""
            CREATE INDEX sentenceTextIndex IF NOT EXISTS 
            FOR (s:Sentence) ON (s.text)
        """)
    print("Index created.")

if __name__ == "__main__":
    createTextIndex()
