import asyncio
import random
from playwright.async_api import async_playwright
from datetime import datetime

class UserAgentRotator:
    """Class to handle user-agent rotation for anti-detection"""
    
    @staticmethod
    def get_random_user_agent():
        """Return a random user-agent string from different browsers and OS"""
        user_agents = [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
            
            # Chrome on Linux
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            
            # Firefox on Linux
            "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
            
            # Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        ]
        return random.choice(user_agents)
    
    @staticmethod
    def get_viewport_size():
        """Return random viewport sizes to simulate different devices"""
        viewports = [
            {"width": 1920, "height": 1080},  # Desktop
            {"width": 1366, "height": 768},   # Laptop
            {"width": 1536, "height": 864},   # Desktop
            {"width": 1440, "height": 900},   # Desktop
            {"width": 1280, "height": 720},   # Desktop/Laptop
        ]
        return random.choice(viewports)

class AntiDetectionBrowser:
    """Class to create browsers with anti-detection features"""
    
    @staticmethod
    async def create_stealth_browser(playwright):
        """Create a browser with anti-detection settings"""
        # Random delays to simulate human behavior
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Get random user-agent and viewport
        user_agent = UserAgentRotator.get_random_user_agent()
        viewport = UserAgentRotator.get_viewport_size()
        
        # Launch browser with additional stealth options
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
                f'--user-agent={user_agent}',
                '--disable-gpu',
                '--disable-software-rasterizer',
            ]
        )
        
        # Create context with additional anti-detection features
        context = await browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            locale='en-US',
            timezone_id='Europe/Dublin',  # Match target website location
            permissions=['geolocation'],
            color_scheme='light',
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
        )
        
        # FIXED: Use add_init_script instead of evaluate_on_new_document
        # Add stealth scripts to hide automation
        await context.add_init_script("""
            // Override navigator properties
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Override chrome property
            Object.defineProperty(navigator, 'chrome', {
                get: () => ({
                    runtime: {},
                    app: {},
                    webstore: {},
                })
            });
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)
        
        page = await context.new_page()
        
        # Additional page-level stealth (optional - can be added here or removed)
        # This is commented out as we're already using context.add_init_script
        # await page.add_init_script("""
        #     Object.defineProperty(navigator, 'webdriver', {
        #         get: () => false
        #     });
        #     
        #     window.chrome = {
        #         runtime: {}
        #     };
        # """)
        
        print(f"Using User-Agent: {user_agent[:80]}...")
        print(f"Viewport size: {viewport['width']}x{viewport['height']}")
        
        return browser, context, page

async def human_like_delay(min_seconds=1, max_seconds=3):
    """Add random delays to simulate human behavior"""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)

async def main():
    async with async_playwright() as p:
        # Create stealth browser with rotated user-agent
        browser, context, page = await AntiDetectionBrowser.create_stealth_browser(p)
        
        try:
            # User inputs for search criteria
            address = input("Enter address (or leave empty for all): ") or "Belmont"
            county = input("Enter county (e.g., Dublin): ") or "Dublin"
            year = input("Enter year (e.g., 2024): ") or "2024"
            start_month = input("Enter start month (01-12, or leave empty for all): ") or ""
            end_month = input("Enter end month (01-12, or leave empty for all): ") or ""
            
            print(f"\nSearching for:")
            print(f"  Address: {address}")
            print(f"  County: {county}")
            print(f"  Year: {year}")
            if start_month:
                print(f"  Start Month: {start_month}")
            if end_month:
                print(f"  End Month: {end_month}")
            print()

            # Navigate to page with delay
            await page.goto("https://www.propertypriceregister.ie/website/npsra/pprweb.nsf/PPR?OpenForm")
            await human_like_delay(2, 4)

            # Accept all cookies if the banner is present
            cookie_accept = page.locator("#cookiescript_accept")
            if await cookie_accept.count() > 0 and await cookie_accept.is_visible():
                print("Accepting cookies...")
                await cookie_accept.click()
                await human_like_delay(1, 2)

            # Fill the form fields with random delays between actions
            print("Filling form...")
            await page.fill("#Address", address)
            await human_like_delay(0.5, 1.5)
            
            await page.select_option("#County", county)
            await human_like_delay(0.5, 1.5)
            
            await page.select_option("#Year", year)
            await human_like_delay(0.5, 1.5)
            
            # Fill start month if provided
            if start_month:
                await page.select_option("#StartMonth", start_month)
                await human_like_delay(0.5, 1.5)
            
            # Fill end month if provided  
            if end_month:
                await page.select_option("#EndMonth", end_month)
                await human_like_delay(0.5, 1.5)

            # Click the 'Perform Search' button with delay
            print("Submitting search...")
            await human_like_delay(1, 2)
            async with page.expect_navigation():
                await page.click("input[value='Perform Search']")
            
            await human_like_delay(2, 4)

            # Wait for the results table to load
            try:
                await page.wait_for_selector("#searchResultsTbl", timeout=15000)
                print("Results table loaded.")
            except:
                # Check if there are no results
                page_content = await page.content()
                if "no results" in page_content.lower() or "no records" in page_content.lower():
                    print("No results found for the search criteria.")
                    return
                else:
                    print("Table not found, but continuing...")
            
            await human_like_delay(1, 2)

            all_results = []
            page_number = 1
            
            while True:
                print(f"\nProcessing page {page_number}...")
                
                # Check if table exists on current page
                if await page.locator("#searchResultsTbl").count() == 0:
                    print("Results table not found on this page.")
                    break
                
                # Extract results from current page
                rows = await page.locator("#searchResultsTbl tbody tr").all()
                
                if len(rows) == 0:
                    print("No rows found in table.")
                    break
                
                # Randomize processing order of rows
                row_indices = list(range(len(rows)))
                if random.random() > 0.5:  # 50% chance to shuffle
                    random.shuffle(row_indices)
                
                for idx in row_indices:
                    row = rows[idx]
                    
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
                    
                    # Small random delay between rows
                    if random.random() > 0.7:  # 30% chance for delay
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                
                # Get the total results info if available
                info_locator = page.locator("#searchResultsTbl_info")
                if await info_locator.count() > 0:
                    info_text = await info_locator.inner_text()
                    print(f"  Found {len(rows)} rows. {info_text}")
                else:
                    print(f"  Found {len(rows)} rows.")
                
                # Check if we're on the last page
                next_button = page.locator("#searchResultsTbl_next")
                
                if await next_button.count() == 0:
                    print("No next button found - stopping.")
                    break
                
                # Check if the button has the 'disabled' class
                is_disabled = await next_button.evaluate('(element) => element.classList.contains("disabled")')
                
                if is_disabled:
                    print("Next button is disabled - reached last page.")
                    break
                else:
                    # Random delay before clicking next
                    await human_like_delay(1, 3)
                    
                    # Random mouse movement simulation before click
                    if random.random() > 0.3:
                        box = await next_button.bounding_box()
                        if box:
                            # Move mouse to random position near button
                            offset_x = random.randint(-10, 10)
                            offset_y = random.randint(-10, 10)
                            await page.mouse.move(box['x'] + box['width']/2 + offset_x, 
                                                box['y'] + box['height']/2 + offset_y)
                            await asyncio.sleep(random.uniform(0.2, 0.5))
                    
                    print("Clicking next button...")
                    await next_button.click()
                    
                    # Variable wait for page load
                    load_delay = random.uniform(1.5, 3.5)
                    await asyncio.sleep(load_delay)
                    
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
                        print("Reached safety limit of 50 pages - stopping.")
                        break
            
            print(f"\n{'='*60}")
            print(f"Total results found: {len(all_results)}")
            
            if all_results:
                # Save to CSV
                import csv
                
                filename = f"property_results_{address[:20]}_{county}_{year}"
                if start_month:
                    filename += f"_from{start_month}"
                if end_month:
                    filename += f"_to{end_month}"
                filename = filename.replace(" ", "_").replace("/", "_") + ".csv"
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Page', 'Date of Sale', 'Price', 'Address']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in all_results:
                        writer.writerow(entry)
                
                print(f"Results saved to '{filename}'")
                
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
                
                print(f"\nProcess completed with {len(all_results)} properties.")
            
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Close browser
            await browser.close()
            print("\nBrowser closed.")

if __name__ == "__main__":
    asyncio.run(main())