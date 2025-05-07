from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# Create Chapter nodes
def create_chapters():
    with driver.session() as session:
        session.run("""
            MATCH (s:Sentence)
            WITH DISTINCT s.chapter AS chapter
            MERGE (:Chapter {id: chapter});
        """)

# Create Source nodes
def create_sources():
    with driver.session() as session:
        session.run("""
            MATCH (s:Sentence)
            WITH DISTINCT s.source AS source
            MERGE (:Source {name: source});
        """)

# Create relationships between Sentence and Chapter
def create_sentence_chapter_relationships():
    with driver.session() as session:
        session.run("""
            MATCH (s:Sentence), (c:Chapter {id: s.chapter})
            MERGE (s)-[:IN_CHAPTER]->(c);
        """)

# Create relationships between Sentence and Source
def create_sentence_source_relationships():
    with driver.session() as session:
        session.run("""
            MATCH (s:Sentence), (src:Source {name: s.source})
            MERGE (s)-[:FROM_SOURCE]->(src);
        """)

# Create relationships between consecutive sentences in the same chapter and source
def create_next_sentence_relationships():
    with driver.session() as session:
        session.run("""
            MATCH (a:Sentence), (b:Sentence)
            WHERE a.chapter = b.chapter AND a.source = b.source AND a.id = b.id - 1
            MERGE (a)-[:NEXT]->(b);
        """)

if __name__ == "__main__":
    create_chapters()
    create_sources()
    create_sentence_chapter_relationships()
    create_sentence_source_relationships()
    create_next_sentence_relationships()
    print("Relationships created successfully!")
