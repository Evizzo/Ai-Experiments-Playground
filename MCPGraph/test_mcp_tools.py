import json
from main import (
    web_search,
    concept_extractor,
    update_graph,
    autoLinkWithGDS,
    query_graph,
    rerank,
    responder,
)

def runFullPipeline(userId, query, prefs):
    # 1) search + extract
    ws = web_search(query)
    ce = concept_extractor(ws)

    # 2) persist + auto-link
    update_graph(userId, ce)
    autoLinkWithGDS()

    # 3) query
    qg = query_graph(userId, prefs)

    # 4) rerank + summarize
    final = rerank({"query_graph": qg, "preferences": prefs})
    summary = responder({
        "web_search": ws,
        "concept_extractor": ce,
        "query_graph": final
    })

    return {
        "summary":     summary,
        "search":      ws,
        "concepts":    ce,
        "graphResults": final
    }

if __name__ == "__main__":
    tests = [
        ("lightweight sports cars with high agility and rear-wheel drive", {"efficiency":"medium","depth":1}),
        ("best petrol track cars under 100k in 2025",              {"maxPrice":100_000,"depth":2}),
        ("trends in performance automotive design and handling in 2025", {"depth":2}),
    ]

    userId = "stef"
    for q, prefs in tests:
        out = runFullPipeline(userId, q, prefs)
        print("\n>> QUERY:", q)
        print(json.dumps(out, indent=2))

    # finally, explain the resulting graph:
    from main import explain_graph
    print("\n>> EXPLAIN_GRAPH:")
    print(explain_graph(userId))
