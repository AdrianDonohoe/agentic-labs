def main():    
    # COMMON IMPORTS
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from langchain.tools import tool

    from pydantic import BaseModel, Field
    from typing_extensions import Annotated

    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    from langchain_core.messages import (HumanMessage, SystemMessage, AIMessage, AnyMessage)

    from langgraph.graph.message import add_messages
    import asyncio
    from playwright.async_api import async_playwright
    load_dotenv()

    class Property(BaseModel):
        address: str = Field(description="The address of the property")
        date: str = Field(description="The date the property was sold")
        price: str = Field(description="The price of the property")

    class PropertyList(BaseModel):
        properties: list[Property] = Field(description="A list of properties sold")


    #TOOLS
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
            def log(message):
                if verbose:
                    print(message)

            async with async_playwright() as p:
                # Simplified browser launch
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = await context.new_page()

                try:
                    log(f"\nSearching for:")
                    log(f"  Address: {address}")
                    log(f"  County: {county}")
                    log(f"  Year: {year}")
                    if start_month:
                        log(f"  Start Month: {start_month}")
                    if end_month:
                        log(f"  End Month: {end_month}")
                    log("")

                    # Navigate to page
                    await page.goto("https://www.propertypriceregister.ie/website/npsra/pprweb.nsf/PPR?OpenForm")
                    await asyncio.sleep(2)

                    # Accept all cookies if the banner is present
                    cookie_accept = page.locator("#cookiescript_accept")
                    if await cookie_accept.count() > 0 and await cookie_accept.is_visible():
                        log("Accepting cookies...")
                        await cookie_accept.click()
                        await asyncio.sleep(1)

                    # Fill the form fields
                    log("Filling form...")
                    await page.fill("#Address", address)
                    await page.select_option("#County", county)
                    await page.select_option("#Year", year)

                    # Fill start month if provided
                    if start_month:
                        await page.select_option("#StartMonth", start_month)

                    # Fill end month if provided  
                    if end_month:
                        await page.select_option("#EndMonth", end_month)

                    # Click the 'Perform Search' button
                    log("Submitting search...")
                    await asyncio.sleep(1)
                    async with page.expect_navigation():
                        await page.click("input[value='Perform Search']")

                    await asyncio.sleep(2)

                    # Wait for the results table to load
                    try:
                        await page.wait_for_selector("#searchResultsTbl", timeout=15000)
                        log("Results table loaded.")
                    except:
                        # Check if there are no results
                        page_content = await page.content()
                        if "no results" in page_content.lower() or "no records" in page_content.lower():
                            log("No results found for the search criteria.")
                            return []
                        else:
                            log("Table not found, but continuing...")

                    await asyncio.sleep(1)

                    all_results = []
                    page_number = 1

                    while True:
                        log(f"\nProcessing page {page_number}...")

                        # Check if table exists on current page
                        if await page.locator("#searchResultsTbl").count() == 0:
                            log("Results table not found on this page.")
                            break
                        
                        # Extract results from current page
                        rows = await page.locator("#searchResultsTbl tbody tr").all()

                        if len(rows) == 0:
                            log("No rows found in table.")
                            break
                        
                        for row in rows:
                            # Get all text content from table cells
                            cols = await row.locator("td").all_text_contents()

                            if len(cols) >= 3:
                                # Try to get address from the link first
                                address_link = row.locator("td:nth-child(3) a")
                                address_text = ""

                                if await address_link.count() > 0:
                                    address_text = await address_link.inner_text()
                                else:
                                    address_text = cols[2]

                                all_results.append({
                                    "Date of Sale": cols[0].strip(),
                                    "Price": cols[1].strip(),
                                    "Address": address_text.strip(),
                                    "Page": page_number
                                })

                        # Get the total results info if available
                        info_locator = page.locator("#searchResultsTbl_info")
                        if await info_locator.count() > 0:
                            info_text = await info_locator.inner_text()
                            log(f"  Found {len(rows)} rows. {info_text}")
                        else:
                            log(f"  Found {len(rows)} rows.")

                        # Check if we're on the last page
                        next_button = page.locator("#searchResultsTbl_next")

                        if await next_button.count() == 0:
                            log("No next button found - stopping.")
                            break
                        
                        # Check if the button has the 'disabled' class
                        is_disabled = await next_button.evaluate('(element) => element.classList.contains("disabled")')

                        if is_disabled:
                            log("Next button is disabled - reached last page.")
                            break
                        else:
                            log("Clicking next button...")
                            await next_button.click()

                            # Wait for page load
                            await asyncio.sleep(2)

                            # Additional wait for table
                            try:
                                await page.wait_for_function(
                                    """() => {
                                        const rows = document.querySelectorAll('#searchResultsTbl tbody tr');
                                        return rows.length > 0;
                                    }""",
                                    timeout=8000
                                )
                            except:
                                await asyncio.sleep(1)

                            page_number += 1
                            if page_number > 50:  # Safety limit
                                log("Reached safety limit of 50 pages - stopping.")
                                break
                            
                    log(f"\n{'='*60}")
                    log(f"Total results found: {len(all_results)}")

                    if all_results and verbose:
                        # Display summary
                        print(f"\nFirst 3 results:")
                        for i, entry in enumerate(all_results[:3]):
                            print(f"{i+1}. {entry['Date of Sale']} - {entry['Price']}")
                            print(f"   {entry['Address'][:60]}...")
                            print()

                        # Statistics
                        try:
                            prices = [
                                float(entry['Price'].replace('€', '').replace(',', '').replace('**', '').strip())
                                for entry in all_results
                            ]
                            print(f"Statistics:")
                            print(f"  Average price: €{sum(prices)/len(prices):,.2f}")
                            print(f"  Price range: €{min(prices):,.2f} - €{max(prices):,.2f}")
                        except:
                            pass
                        
                    return all_results

                except Exception as e:
                    if verbose:
                        print(f"Error occurred: {e}")
                        import traceback
                        traceback.print_exc()
                    return []

                finally:
                    # Close browser
                    await browser.close()
                    if verbose:
                        print("\nBrowser closed.")

        # Synchronous entry point that runs the async function
        return asyncio.run(_async_scrape())

    # code execution tool
    import contextlib
    import io

    @tool
    def execute_code(code: str = Field(description="Code to be executed.")) -> str:
        """Executes the given Python code and returns the output or error message.
            You have access to the pandas library."""
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code)
            result = f.getvalue()    
        except Exception as e:
                result = f"Error: {e}"
        return result



    # MAIN GRAPH STATE
    class ScrapingState(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages]
        properties: PropertyList | None = None

    # Property Data Search Subgraph state
    class SubgraphState(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages]
        properties: PropertyList | None = None

    #MODELS
    llm = ChatGoogleGenerativeAI(
        # model=os.getenv("GOOGLE_API_MODEL"),
        model="gemini-2.5-flash",
        temperature=0)

    structure_model = llm.with_structured_output(PropertyList)


    # STATE
    class ScrapingState(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages]
        properties: PropertyList | None = None

    #SUBGRAPH NODES
    tools = [execute_code]
    tool_llm = llm.bind_tools(tools)

    def generate_code_node(state: SubgraphState) -> dict:
        # Bind the tool to the llm

        # Debug the messages being sent
        print(f"DEBUG - Messages being sent to LLM: {state.messages}")

        # Invoke the llm
        result = tool_llm.invoke(state.messages)
        print(f"Generate Code Node Result: {result}")
        print(f"DEBUG - Result type: {type(result)}")
        print(f"DEBUG - Result content: {result.content}")
        print(f"DEBUG - Result additional_kwargs: {result.additional_kwargs}")

        return {"messages": result}

    #COMPILE SUBGRAPH

    data_analysis_graph = StateGraph(SubgraphState)
    data_analysis_graph.add_node("code", generate_code_node)
    data_analysis_graph.add_node("execute", ToolNode(tools=[execute_code]))

    data_analysis_graph.add_edge(START, "code")
    data_analysis_graph.add_conditional_edges("code", lambda state: state.messages[-1].tool_calls == [],
                                path_map={True: END, False: "execute"})
    data_analysis_graph.add_edge("execute", "code")


    data_analysis_graph = data_analysis_graph.compile()


    # MAIN GRAPH NODES
    def structure_node(state: ScrapingState) -> dict:

        msgs = [HumanMessage(content="Structure the following property data into a PropertyList format:\n" + str(state.messages[-1].content))]

        response = structure_model.invoke(msgs)

        return {"properties": response}

    def scrape_property_data_node(state: ScrapingState) -> dict:

        result = llm.bind_tools(tools=[scrape_property_data]).invoke(state.messages)
        #print(f"Scrape Node Result: {result.messages[-1]}")
        return {"messages": state.messages + [result]}

    # Entrance to property data subgraph
    def property_data_node(state: ScrapingState) -> dict:
        # Only if we have properties
        if not state.properties or not state.properties.properties:
            return {"messages": [AIMessage(content="No properties available.")]}
        else:
            # INSTRUCTIONS
            system_prompt = """You are an expert Python programmer and data scientist. Your task is to analyse the provided property data.
            """ 

            # Convert properties to string
            properties_data = ""
            if state.properties and state.properties.properties:
                properties_data = "\nProperties to analyze:\n"
                for i, prop in enumerate(state.properties.properties):
                    properties_data += f"{i+1}. Address: {prop.address}, Date: {prop.date}, Price: {prop.price}\n"

            instruction_message = f"""Write Python code to analyze the provided property data using pandas.

            1. Create a pandas DataFrame from the properties provided
            2. Clean the data (convert price to numeric, parse dates)
            3. Compute statistics (average, min, max, count)

            \n\n{properties_data}"""



            #print(f"USER PROMPT:\n{user_prompt}\n")

            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=instruction_message)
            ]
            subgraph_state = SubgraphState(messages=messages, properties=state.properties)
            # Transform the state to the subgraph state
            subgraph_output = data_analysis_graph.invoke(subgraph_state)  
            # Transform response back to the parent state

            return {"messages": subgraph_output['messages']}



    

    # MAIN GRAPH
    graph = StateGraph(ScrapingState)
    graph.add_node("scrape", scrape_property_data_node)
    graph.add_node("tool_node", ToolNode(tools=[scrape_property_data]))#
    graph.add_node("structured_node", structure_node)
    graph.add_node("property_data", property_data_node)

    graph.add_conditional_edges("scrape", 
        lambda state: len(state.messages[-1].tool_calls) > 0 if hasattr(state.messages[-1], 'tool_calls') else False,
        path_map={True: "tool_node", False: "structured_node"}
    )


    graph.add_edge(START, "scrape")
    graph.add_edge("tool_node", "scrape")
    graph.add_edge("structured_node", "property_data")
    graph.add_edge("property_data", END)




    scrape_graph = graph.compile()

    # INSTRUCTIONS
    system_message = """
    You are an expert Python web scraper. 
    """
    instruction_message = """
    You are required to scrape data from the Irish property price register. This website is a registry of property prices in Ireland which can be searched by various criteria.

    Scrape using the {address} as the Address, {county} as the County, {year} as the Year, {smonth} as the Start Month, {emonth} as the End Month.

    Final Output:
    - Return a list of properties with their address, date of sale and price.
    """

    messages = [    SystemMessage(content=system_message),
                    HumanMessage(content=instruction_message.format(address="Parkview", county="Dublin", year="2024", smonth="", emonth="")) ]

    result = scrape_graph.invoke(ScrapingState(messages=messages))
    print(result['messages'][-1].content)
    print(result)


if __name__ == "__main__":
    main()