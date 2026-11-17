"""R2: print-then-block inside asyncio.run; SIGINT should NOT kill promptly (cancellation
can't reach a task blocked in synchronous recv), so a later SIGTERM kills unflushed."""
import asyncio, requests, sys
from trickle_server import serve

async def main():
    print("STARTED: about to block in synchronous recv")  # goes to block buffer only
    requests.get("http://127.0.0.1:18741/", timeout=60)
    print("FINISHED (should not happen)")

if __name__ == "__main__":
    serve()
    asyncio.run(main())
