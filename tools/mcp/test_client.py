#!/usr/bin/env python3
"""Minimal scripted MCP client: launches server.py over stdio, calls
selfcheck(), and asserts an overall PASS verdict against the pinned
SmolLM2-135M v0.3b vector. Run: uv run test_client.py"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable, args=[str(HERE / "server.py")]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            result = await session.call_tool("selfcheck", {})
            text = result.content[0].text
            data = json.loads(text)
            print(json.dumps(data, indent=2)[:4000])

            verdict = data.get("verdict")
            print(f"\nVERDICT: {verdict}")
            return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
