import re
import datetime

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

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


def find_vector_hits(query, limit=5):
    emb = model.encode(query).tolist()
    with driver.session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('sentence_embedding_index', $limit, $embedding)
            YIELD node, score
            RETURN node.content AS text, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            embedding=emb, limit=limit
        )
        return [(r["text"], r["score"]) for r in result]


def find_keyword_hits(query, limit=5):
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return []
    kw = tokens[-1]
    with driver.session() as session:
        result = session.run(
            "MATCH (t:Text) WHERE toLower(t.content) CONTAINS toLower($kw) "
            "RETURN t.content AS text LIMIT $limit", kw=kw, limit=limit
        )
        return [r["text"] for r in result]


def generate_vector_answer(question):
    hits = find_vector_hits(question)
    context = "\n".join([text for text, _ in hits])
    prompt = f"Context (vector-only):\n{context}\n\nAnswer: {question}"
    return gemini.invoke(prompt).content.strip(), hits


def generate_graph_answer(question):
    hits = find_keyword_hits(question)
    context = "\n".join(hits)
    prompt = f"Context (keyword-only):\n{context}\n\nAnswer: {question}"
    return gemini.invoke(prompt).content.strip(), hits


def generate_hybrid_answer(question):
    vec_hits = find_vector_hits(question)
    graph_ctx = []
    with driver.session() as session:
        for text, _ in vec_hits:
            rec = session.run(
                "MATCH (sec:Section)-[:INCLUDES]->(t:Text {content: $txt})"
                " MATCH (sec)<-[:CONTAINS]-(chap:Chapter)"
                " MATCH (sec)-[:INCLUDES]->(u:Text)"
                " RETURN u.content AS text LIMIT 5",
                txt=text
            )
            graph_ctx.extend([r["text"] for r in rec])
    unique = list(dict.fromkeys([t for t,_ in vec_hits] + graph_ctx))
    context = "\n".join(unique)
    prompt = f"Context (hybrid):\n{context}\n\nAnswer: {question}"
    return gemini.invoke(prompt).content.strip(), vec_hits, graph_ctx


if __name__ == "__main__":
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("compareSearches", f"logs-{ts}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "results.txt")

    def out(line: str = "", end: str = "\n"):
        print(line, end=end)
        log_file.write(line + end)

    tests = [
        "Who is the Doctor?",
        "What is the TARDIS?",
        "What decision did the Doctor face?",
        "Describe the Zygon invasion.",
        "How does the Doctor travel through time?"
    ]

    with open(log_path, "w") as log_file:
        out(f"Logging results to {log_path}\n")

        for q in tests:
            out(f"\n=== Question: {q} ===\n")

            # Vector-only
            vec_ans, vec_hits = generate_vector_answer(q)
            out("-- Vector-only Retrieval --")
            for i,(txt,score) in enumerate(vec_hits,1):
                out(f"{i}. ({score:.3f}) {txt}")
            out(f"Answer: {vec_ans}\n")

            # Graph-only (keyword)
            g_ans, g_hits = generate_graph_answer(q)
            out("-- Graph-only Retrieval --")
            for i,txt in enumerate(g_hits,1):
                out(f"{i}. {txt}")
            out(f"Answer: {g_ans}\n")

            # Hybrid
            h_ans, v_hits, g_ctx = generate_hybrid_answer(q)
            out("-- Hybrid Retrieval --")
            for i,(txt,score) in enumerate(v_hits,1):
                out(f"V{i}. ({score:.3f}) {txt}")
            for i,txt in enumerate(g_ctx[:5],1):
                out(f"G{i}. {txt}")
            out(f"Answer: {h_ans}")
            out("=============\n")

    print(f"\n✅ Done—see full logs in {log_path}")