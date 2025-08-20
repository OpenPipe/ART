#!/usr/bin/env python3

import asyncio

from just_the_facts.scenarios import train_urls, val_urls
from just_the_facts.utils import scrape_article


async def test_scraper():
    """Test the scrape_article function with example URLs"""

    # Test URLs from different news sources (using homepage URLs that should exist)
    test_urls = train_urls + val_urls

    for url in test_urls:
        try:
            print(f"\nTesting URL: {url}")
            article_text = await scrape_article(url)
            print(f"Successfully scraped {len(article_text)} characters")
            print(f"First 200 characters: {article_text[:200]}...")
        except Exception as e:
            print(f"Failed to scrape {url}: {str(e)}")
            raise e


if __name__ == "__main__":
    asyncio.run(test_scraper())
