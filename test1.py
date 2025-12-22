import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")
        await page.goto("https://www.propertypriceregister.ie/website/npsra/pprweb.nsf/PPR?OpenForm")
        await page.wait_for_timeout(5000)

        # Accept all cookies if the banner is present
        if await page.locator("#cookiescript_accept").is_visible():
            await page.click("#cookiescript_accept")
            await page.wait_for_timeout(2000)

        # Fill the form fields
        await page.fill("#Address", "Belmont")
        await page.select_option("#County", "Dublin")
        await page.select_option("#Year", "2024")

        # Click the 'Perform Search' button
        async with page.expect_navigation():
            await page.click("input[value='Perform Search']")

        # Wait for the results table to load
        await page.wait_for_selector("#searchResultsTbl")
        await page.wait_for_timeout(2000)  # Allow table to fully render

        # Extract the results from the table
        results = []
        rows = await page.locator("#searchResultsTbl tbody tr").all()
        
        for row in rows:
            # Get all text content from table cells
            cols = await row.locator("td").all_text_contents()
            
            if len(cols) >= 3:  # We need at least 3 columns
                # Try to get address from the link first
                address_link = row.locator("td:nth-child(3) a")
                address_text = ""
                
                # Check if the link exists
                if await address_link.count() > 0:
                    address_text = await address_link.inner_text()
                else:
                    address_text = cols[2]  # Fallback to raw text from the cell
                
                results.append({
                    "Date of Sale": cols[0].strip(),
                    "Price": cols[1].strip(),
                    "Address": address_text.strip()
                })
        
        print(f"Number of results found: {len(results)}")
        print("\nFirst five entries:")
        for i, entry in enumerate(results[:5]):
            print(f"Entry {i+1}: {entry}")
        
        # Print all results
        print("\nAll results:")
        for i, entry in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Date of Sale: {entry['Date of Sale']}")
            print(f"  Price: {entry['Price']}")
            print(f"  Address: {entry['Address']}")
            print()

        await browser.close()

asyncio.run(main())