from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import json, os
from dotenv import load_dotenv

load_dotenv() 

URI      = os.getenv("NEO4J_URI")
USER     = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

with open('input/data.json', 'r') as f:
    data = json.load(f)

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
model  = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def clearAll(tx):
    # detach-delete EVERYTHING
    tx.run("MATCH (n) DETACH DELETE n")

def createIndexes(tx):
    # simple property indexes
    tx.run("CREATE INDEX IF NOT EXISTS FOR (b:Book)    ON (b.title)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:Chapter) ON (c.number)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.number)")
    # vector index on Text.embedding
    tx.run("""
      CREATE VECTOR INDEX sentence_embedding_index IF NOT EXISTS
        FOR (t:Text) ON (t.embedding)
        OPTIONS { indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }}
    """)

def insertItem(tx, item):
    # 1) Book node
    tx.run("""
      MERGE (b:Book {
        title: $book_title,
        author: $author,
        publication_year: $year
      })
    """, book_title=item['book_title'],
         author=item['author'],
         year=item['publication_year'])

    # 2) Chapter node + relationship
    tx.run("""
      MATCH (b:Book {title:$book_title})
      MERGE (c:Chapter {
        number: $chapter,
        title:  $chapter_title
      })
      MERGE (b)-[:HAS]->(c)
    """, book_title=item['book_title'],
         chapter=item['chapter'],
         chapter_title=item['chapter_title'])

    # 3) Section node + relationship
    tx.run("""
      MATCH (c:Chapter {number:$chapter, title:$chapter_title})
      MERGE (s:Section {number: $section})
      MERGE (c)-[:CONTAINS]->(s)
    """, chapter=item['chapter'],
         chapter_title=item['chapter_title'],
         section=item['section'])

    # 4) Text node + embedding + relationship
    embedding = model.encode(item['text']).tolist()
    tx.run("""
      MATCH (s:Section {number:$section})
      MERGE (t:Text {
        content:   $text,
        embedding: $embedding
      })
      MERGE (s)-[:INCLUDES]->(t)
    """, section=item['section'],
         text=item['text'],
         embedding=embedding)

def main():
    with driver.session() as session:
        # **fresh start**:
        session.execute_write(clearAll)
        # build indexes (Book.title, Chapter.number, Section.number, Text.embedding)
        session.execute_write(createIndexes)

        # ingest every JSON record
        for item in data:
            session.execute_write(insertItem, item)

    print("Done: cleared DB, built indexes, and ingested data with embeddings.")

if __name__ == "__main__":
    main()
