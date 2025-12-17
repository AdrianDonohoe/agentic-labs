from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time_srv")

@mcp.tool()
def get_current_time():
    """Get the current time in ISO 8601 format."""
    from datetime import datetime
    return datetime.now().isoformat()

@mcp.resource("config://settings")
def get_settings() -> str:
    """Get application settings."""
    return """{
  "theme": "dark",
  "language": "en",
  "debug": false
}"""

@mcp.tool()
def get_timezone_time(timezone: str) -> str:
    """Get the current time in ISO 8601 format."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(timezone)).isoformat()

@mcp.resource("file://documents/{name}")
def read_document(name: str) -> str:
    """Read a document by name."""
    # This would normally read from disk
    return f"Content of {name}"

def main():
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
