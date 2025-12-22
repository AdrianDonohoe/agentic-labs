# Import patch first
import langchain_patch

# COMMON IMPORTS
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import ( HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage)

from langgraph.graph.message import add_messages
import asyncio, random, os
from playwright.async_api import async_playwright
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Define Property and PropertyList models
class Property(BaseModel):
    address: str = Field(description="The address of the property")
    date: str = Field(description="The date the property was sold")
    price: str = Field(description="The price of the property")
    type: Optional[str] = Field(default=None, description="The type of the property (e.g., apartment, semi-detached, detached, terraced house etc.)")
    bedrooms: Optional[int] = Field(default=None, description="The number of bedrooms in the property")
    size_sq_m: Optional[float] = Field(default=None, description="The size of the property in square meters")

class PropertyList(BaseModel):
    properties: list[Property] = Field(description="A list of properties sold")

# UserAgentRotator and AntiDetectionBrowser classes remain the same
class UserAgentRotator:
    @staticmethod
    def get_random_user_agent():
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        ]
        return random.choice(user_agents)
    
    @staticmethod
    def get_viewport_size():
        viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
        ]
        return random.choice(viewports)

class AntiDetectionBrowser:
    @staticmethod
    async def create_stealth_browser(playwright):
        """Create a browser with anti-detection settings"""
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        user_agent = UserAgentRotator.get_random_user_agent()
        viewport = UserAgentRotator.get_viewport_size()
        
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                f'--user-agent={user_agent}',
                '--disable-gpu',
            ]
        )
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            locale='en-US',
            timezone_id='Europe/Dublin',
        )
        
        page = await context.new_page()
        
        return browser, context, page

async def human_like_delay(min_seconds=1, max_seconds=3):
    """Add random delays to simulate human behavior"""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)

# Tools remain the same
@tool
def scrape_property_data(address="Belmont", county="Dublin", year="2024", 
                          start_month=None, end_month=None, verbose=False):
    """
    Scrape Irish property price data from the Property Price Register.
    
    Args:
        address (str): Address to search for (default: "Belmont")
        county (str): County to search in (default: "Dublin")
        year (str): Year to search (default: "2024")
        start_month (str, optional): Start month in "01"-"12" format
        end_month (str, optional): End month in "01"-"12" format
        verbose (bool): Whether to print progress messages (default: False)
    
    Returns:
        list: List of dictionaries containing property data, or empty list if no results
    """
    async def _async_scrape():
        """Async implementation of property data scraping"""
        def log(message):
            if verbose:
                print(message)
        
        async with async_playwright() as p:
            browser, context, page = await AntiDetectionBrowser.create_stealth_browser(p)
            
            try:
                # Your scraping logic here
                log(f"Scraping data for {address} in {county}")
                await page.goto("https://www.propertypriceregister.ie")
                await human_like_delay(2, 4)
                
                # Simplified scraping logic - you should keep your full implementation
                return [
                    {
                        "Date of Sale": "2024-01-15",
                        "Price": "€450,000",
                        "Address": "22 The Cedar, Parkview, Stepaside Dublin 18"
                    },
                    {
                        "Date of Sale": "2024-02-20",
                        "Price": "€520,000",
                        "Address": "15 Oak Lane, Dublin 18"
                    }
                ]
                    
            except Exception as e:
                if verbose:
                    print(f"Error occurred: {e}")
                return []
                
            finally:
                await browser.close()
    
    return asyncio.run(_async_scrape())

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0)

structure_model = llm.with_structured_output(PropertyList)

# STATE for main graph
class ScrapingState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    properties: PropertyList | None = None
    enriched_properties: PropertyList | None = None
    property_details_cache: Dict[str, Dict[str, Any]] = {}

# STATE for property detail search subgraph
class PropertyDetailState(BaseModel):
    property_address: str
    messages: Annotated[list[AnyMessage], add_messages]
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    size_sq_m: Optional[float] = None

# NODES for main graph
def scrape_property_data_node(state: ScrapingState) -> dict:
    result = llm.bind_tools(tools=[scrape_property_data]).invoke(state.messages)
    return {"messages": result}

def structure_node(state: ScrapingState) -> dict:
    msgs = [HumanMessage(content="Structure the following property data into a PropertyList format:\n" + str(state.messages[-1].content))]
    response = structure_model.invoke(msgs)
    return {"properties": response}

# Property detail search subgraph
async def property_detail_search_node(state: PropertyDetailState) -> dict:
    """Search for property details using Tavily"""
    # Initialize Tavily client for property detail search
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    tavily_server = MultiServerMCPClient(
        {
            "tavily_srv": {
                "transport": "streamable_http",
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
            }
        })
    
    tools = await tavily_server.get_tools()
    
    # Create search query for the specific property
    search_query = f"Property details for: {state.property_address}. Search myhome.ie and daft.ie for property type, number of bedrooms, and size in square meters."
    
    messages = [
        SystemMessage(content="You are a helpful assistant that searches for property details in Ireland. Search myhome.ie and daft.ie specifically."),
        HumanMessage(content=search_query)
    ]
    
    # Create a simple graph for the property detail search
    class SearchState(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages]
    
    async def action(state: SearchState) -> dict:
        """Execute search action"""
        result = await llm.ainvoke(state.messages)
        return {"messages": [result]}
    
    def summary_node(state: SearchState) -> dict:
        """Summarizes the search results."""
        messages = [SystemMessage(content="You are a helpful assistant that summarizes internet search results."),
                    HumanMessage(content=f"Summarize the following search results for the query. Return the property type and number of bedrooms and size in square meters :\n {state.messages[-1].content}")]
        try:
            summary = llm.invoke(messages)
            return {"messages": [summary]}
        except Exception as e:
            error_message = f"Error(summary_node) : {e}"
            return {"messages": [AIMessage(content=error_message)]}
    
    # Create the search graph
    search_graph = StateGraph(SearchState)
    search_graph.add_node("action", action)
    search_graph.add_node("tools", ToolNode(tools))
    search_graph.add_node("summary", summary_node)
    
    search_graph.add_edge(START, "action")
    
    def route_after_action(state: SearchState):
        if state.messages and state.messages[-1].tool_calls:
            return "tools"
        return "summary"
    
    search_graph.add_conditional_edges("action", route_after_action)
    search_graph.add_edge("tools", "action")
    search_graph.add_edge("summary", END)
    
    compiled_search_graph = search_graph.compile()
    
    # Run the search graph
    search_result = await compiled_search_graph.ainvoke(
        SearchState(messages=messages)
    )
    
    return {"messages": search_result["messages"]}

def extract_property_details_node(state: PropertyDetailState) -> dict:
    """Extract property details from search results"""
    try:
        # Extract property details from search results
        messages = [
            SystemMessage(content="Extract ONLY the property type (e.g., apartment, semi-detached, detached, terraced), number of bedrooms (as integer), and size in square meters (as float) from the search results. If information is not found, return None for that field."),
            HumanMessage(content=f"Extract property details from: {state.messages[-1].content}")
        ]
        
        # Use structured output for extraction
        class PropertyDetails(BaseModel):
            property_type: Optional[str] = None
            bedrooms: Optional[int] = None
            size_sq_m: Optional[float] = None
        
        extraction_model = llm.with_structured_output(PropertyDetails)
        details = extraction_model.invoke(messages)
        
        return {
            "property_type": details.property_type,
            "bedrooms": details.bedrooms,
            "size_sq_m": details.size_sq_m
        }
    except Exception as e:
        print(f"Error extracting property details: {e}")
        return {}

# Create the property detail search subgraph
def create_property_detail_subgraph():
    """Create and return a compiled property detail search subgraph"""
    property_detail_graph = StateGraph(PropertyDetailState)
    property_detail_graph.add_node("search", property_detail_search_node)
    property_detail_graph.add_node("extract", extract_property_details_node)
    
    property_detail_graph.add_edge(START, "search")
    property_detail_graph.add_edge("search", "extract")
    property_detail_graph.add_edge("extract", END)
    
    return property_detail_graph.compile()

# Function to enrich a single property
async def enrich_single_property(property_address: str) -> Dict[str, Any]:
    """Run subgraph to enrich a single property with details"""
    subgraph = create_property_detail_subgraph()
    
    try:
        result = await subgraph.ainvoke(
            PropertyDetailState(
                property_address=property_address,
                messages=[]
            )
        )
        
        return {
            "type": result.get("property_type"),
            "bedrooms": result.get("bedrooms"),
            "size_sq_m": result.get("size_sq_m")
        }
        
    except Exception as e:
        print(f"Error enriching property {property_address}: {e}")
        return {}

def enrich_properties_node(state: ScrapingState) -> dict:
    """Enrich properties with details using subgraph for each property"""
    if not state.properties or not state.properties.properties:
        return {"enriched_properties": None}
    
    enriched_properties = []
    
    # Process properties
    for i, prop in enumerate(state.properties.properties):
        print(f"Processing property {i+1}/{len(state.properties.properties)}: {prop.address[:50]}...")
        
        # Check cache first
        if prop.address in state.property_details_cache:
            details = state.property_details_cache[prop.address]
        else:
            # Run subgraph for this property
            details = asyncio.run(enrich_single_property(prop.address))
            # Cache the results
            if state.property_details_cache is None:
                state.property_details_cache = {}
            state.property_details_cache[prop.address] = details
        
        # Create enriched property
        enriched_property = Property(
            address=prop.address,
            date=prop.date,
            price=prop.price,
            type=details.get("type"),
            bedrooms=details.get("bedrooms"),
            size_sq_m=details.get("size_sq_m")
        )
        enriched_properties.append(enriched_property)
    
    # Create enriched PropertyList
    enriched_property_list = PropertyList(properties=enriched_properties)
    
    return {
        "enriched_properties": enriched_property_list,
        "property_details_cache": state.property_details_cache
    }

# MAIN GRAPH with subgraph integration
graph = StateGraph(ScrapingState)
graph.add_node("scrape", scrape_property_data_node)
graph.add_node("tool_node", ToolNode(tools=[scrape_property_data]))
graph.add_node("structured_node", structure_node)
graph.add_node("enrich_properties", enrich_properties_node)

# Conditional routing based on tool calls
def route_after_scrape(state: ScrapingState):
    if state.messages and state.messages[-1].tool_calls:
        return "tool_node"
    return "structured_node"

graph.add_edge(START, "scrape")
graph.add_conditional_edges("scrape", route_after_scrape)
graph.add_edge("tool_node", "scrape")
graph.add_edge("structured_node", "enrich_properties")
graph.add_edge("enrich_properties", END)

# Compile the main graph
scrape_graph = graph.compile()

# Example usage
async def main():
    system_message = "You are an expert Python web scraper."
    instruction_message = """
    You are required to scrape data from the Irish property price register.
    Scrape using the {address} as the Address, {county} as the County, 
    {year} as the Year, {smonth} as the Start Month, {emonth} as the End Month.
    Final Output:
    - Return a list of properties with their address, date of sale and price.
    """

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=instruction_message.format(
            address="Parkview", 
            county="Dublin", 
            year="2024", 
            smonth="", 
            emonth=""
        ))
    ]

    # Run the main graph
    result = await scrape_graph.ainvoke(ScrapingState(messages=messages))
    
    # Display results
    if result.get("enriched_properties"):
        print("\n" + "="*60)
        print("ENRICHED PROPERTIES")
        print("="*60)
        for i, prop in enumerate(result["enriched_properties"].properties, 1):
            print(f"\nProperty {i}:")
            print(f"  Address: {prop.address}")
            print(f"  Date: {prop.date}")
            print(f"  Price: {prop.price}")
            print(f"  Type: {prop.type or 'N/A'}")
            print(f"  Bedrooms: {prop.bedrooms or 'N/A'}")
            print(f"  Size (sq m): {prop.size_sq_m or 'N/A'}")
    elif result.get("properties"):
        print("\n" + "="*60)
        print("ORIGINAL PROPERTIES (not enriched)")
        print("="*60)
        for i, prop in enumerate(result["properties"].properties, 1):
            print(f"\nProperty {i}:")
            print(f"  Address: {prop.address}")
            print(f"  Date: {prop.date}")
            print(f"  Price: {prop.price}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())