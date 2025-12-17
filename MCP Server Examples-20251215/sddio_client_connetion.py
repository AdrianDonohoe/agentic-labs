from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run","C:/Users/Andrew/Documents/dkit-projects/agentic-labs/MCP/get-time.py"],
        }
    }
)

async def get_tools():
    return await client.get_tools()   

def main():
    import asyncio
    tools = asyncio.run(get_tools())
    print(tools)

if __name__ == "__main__":
    main()
    