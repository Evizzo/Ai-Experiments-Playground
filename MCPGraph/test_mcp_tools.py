import asyncio
from pprint import pprint
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    server_params = StdioServerParameters(
        command="python",
        args=["main.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools.tools])

            tests = [
                ("lightweight sports cars with high agility and rear-wheel drive", {"efficiency": "medium", "depth": 1}),
                ("best petrol track cars under 100k in 2025", {"maxPrice": 100_000, "depth": 2}),
                ("trends in performance automotive design and handling in 2025", {"depth": 2}),
            ]

            user_id = "stef"
            for q, prefs in tests:
                print("\n>> QUERY:", q)
                result = await session.call_tool("fullPipeline", {
                    "userId": user_id,
                    "query": q,
                    "preferences": prefs
                })
                pprint(getattr(result, "content", result))

            print("\n🔍 EXPLAIN_GRAPH:")
            explain = await session.call_tool("explain_graph", {"userId": user_id})
            pprint(getattr(explain, "content", explain))

if __name__ == "__main__":
    asyncio.run(run())
