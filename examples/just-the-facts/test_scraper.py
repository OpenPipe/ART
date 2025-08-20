#!/usr/bin/env python3

import asyncio

from just_the_facts.utils import scrape_article


async def test_scraper():
    """Test the scrape_article function with example URLs"""

    # Test URLs from different news sources (using homepage URLs that should exist)
    test_urls = [
        # Trump wars
        "https://www.bbc.com/news/articles/c5y3599gx4qo",
        # Australia - Netanyahu
        "https://www.reuters.com/world/asia-pacific/australias-albanese-downplays-netanyahus-criticism-ties-sour-2025-08-20/",
        # Trump weaponizing justice system
        "https://www.foxnews.com/politics/schiff-launches-legal-defense-fund-response-claims-trump-weaponizing-justice-system",
    ]

    for url in test_urls:
        try:
            print(f"\nTesting URL: {url}")
            article_text = await scrape_article(url)
            print(f"Successfully scraped {len(article_text)} characters")
            print(f"First 200 characters: {article_text[:200]}...")
        except Exception as e:
            print(f"Failed to scrape {url}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_scraper())
