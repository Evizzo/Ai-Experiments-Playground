
# this one should be ignored

# this one should be ignored

# this one should be ignored

from queryGraphWithEmbedding import model
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

gemini = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

def hybridSearchAndAnswer(question, limit=5):
    embedding = model.encode(question).tolist()

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('sentence_embedding_index', $limit, $embedding)
            YIELD node, score
            RETURN node
        """, embedding=embedding, limit=limit)

        topNodes = [r["node"] for r in result]

    context = []
    with driver.session() as session:
        for node in topNodes:
            nodeId = node.id
            subResult = session.run("""
                MATCH (s:Sentence)
                WHERE id(s) = $id
                OPTIONAL MATCH (s)<-[:HAS]-(c:Chapter)-[:HAS]->(related:Sentence)
                RETURN s.text AS self, collect(DISTINCT related.text) AS related
            """, id=nodeId)
            for r in subResult:
                context.append(r["self"])
                context.extend(r["related"])

    uniqueContext = list(dict.fromkeys(context))[:limit]
    contextBlock = "\n".join(uniqueContext)

    prompt = f"Context:\n{contextBlock}\n\nAnswer the question: {question}"
    response = gemini.invoke(prompt)
    return response.content.strip()

def parseFilters(rawInput):
    filters = {}
    for part in rawInput.split(","):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            filters[k.strip()] = v.strip()
    return filters

if __name__ == "__main__":
    print("Ignore this code/🤖 Ask anything! (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        try:
            a = hybridSearchAndAnswer(q)
            print("\n🧠 Gemini says:\n", a)
        except Exception as e:
            print("❌ Error:", e)
