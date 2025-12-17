import asyncio
from playwright.async_api import async_playwright

async def main():
      async with async_playwright() as p:
          browser = await p.chromium.launch(headless=True)
          page = await browser.new_page(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")
          await page.goto("https://www.propertypriceregister.ie/website/npsra/pprweb.nsf/PPR?OpenForm")
          await page.wait_for_timeout(5000)          
          

# Extract results
          results = await page.locator('.resultsDiv table tbody tr').all()
          
          extracted_results = []
          for result in results:
              columns = await result.locator('td').all_text_contents()
              if len(columns) >= 4:
                  extracted_results.append({
                      'Address': columns[0],
                      'County': columns[1],
                      'Price': columns[2],
                      'Date of Sale': columns[3]
                  })
          
          num_results = len(extracted_results)
          print(f"Number of results found: {num_results}")
          print("First five entries:")
          for i, entry in enumerate(extracted_results[:5]):
              print(f"Entry {i+1}: {entry}")

          await browser.close()

asyncio.run(main())