from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
model = SentenceTransformer("all-MiniLM-L6-v2")

gemini = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

def findSimilarSentences(query, limit=5):
    emb = model.encode(query).tolist()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('sentence_embedding_index', $limit, $embedding)
            YIELD node, score
            WITH node, score
            MATCH (sec:Section)-[:INCLUDES]->(node)
            MATCH (chap:Chapter)-[:CONTAINS]->(sec)
            RETURN node.content AS text,
                   chap.title   AS chapter,
                   sec.number   AS section,
                   score
            ORDER BY score DESC
        """, embedding=emb, limit=limit)
        hits = [(r["text"], r["chapter"], r["section"], r["score"]) for r in result]

    print("🔍 [vector hits]:", [(chap,sec,round(s,3)) for _,chap,sec,s in hits])
    return hits

def findRelatedContent(chapter, section):
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chapter)-[:CONTAINS]->(s:Section)-[:INCLUDES]->(t:Text)
            WHERE c.title = $chapter AND s.number = $section
            RETURN t.content AS text
        """, chapter=chapter, section=section)
        related = [r["text"] for r in result]

    print(f"↳ [graph adds for Ch={chapter!r}, Sec={section}]:", related)
    return related

def generateAnswer(question):
    # 1) Vector step
    hits = findSimilarSentences(question)
    vector_count = len(hits)

    # 2) Graph step
    graph_items = []
    for text, chap, sec, _ in hits:
        graph_items.extend(findRelatedContent(chap, sec))
    graph_count = len(graph_items)

    # 3) Build unique context
    all_context = [h[0] for h in hits] + graph_items
    unique_context = list(dict.fromkeys(c for c in all_context if isinstance(c, str)))

    print(f"✅ Hybrid search confirmed: {vector_count} vector hits + {graph_count} graph expansions ({len(unique_context)} total unique snippets)\n")

    # 4) Ask Gemini
    ctx_block = "\n".join(unique_context)
    prompt = f"Context:\n{ctx_block}\n\nAnswer the question: {question}"
    return gemini.invoke(prompt).content.strip()

if __name__ == "__main__":
    print("\n🔎 Write your question here (type 'exit' to quit):\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        hits = findSimilarSentences(q)
        print("\n📌 Top Matches:")
        for text, chap, sec, score in hits:
            print(f"  Ch{chap}·Sec{sec} ({score:.3f}): {text}")

        answer = generateAnswer(q)
        print("\n🧠 Gemini says:\n", answer, "\n")
