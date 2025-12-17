

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time")

@mcp.tool()
def get_current_time():
    """Get the current time in ISO 8601 format."""
    from datetime import datetime
    return datetime.now().isoformat()
    
    


if __name__ == "__main__":
    mcp.run(transport="stdio")