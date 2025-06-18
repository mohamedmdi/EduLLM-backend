# run_query.py
import asyncio
from rag_engine import answer_query

async def main():
    query = "What is overfitting in machine learning and how can we prevent it?"
    print("=== Answer ===")
    async for part in answer_query(query):
        print(part, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
