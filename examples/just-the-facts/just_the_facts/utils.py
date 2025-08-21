import asyncio

import aiohttp
from newspaper import Article

cached_articles = {}


async def scrape_article(url: str) -> str:
    """
    Scrape article text from a news URL (Fox News, MSNBC, etc.)

    Args:
        url: The URL of the news article to scrape

    Returns:
        The cleaned article text as a string

    Raises:
        Exception: If article cannot be scraped or processed
    """
    if url in cached_articles:
        return cached_articles[url]

    try:
        # Use newspaper3k to download and parse the article
        article = Article(url)

        # Configure user agent and headers to avoid blocks
        article.config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        article.config.request_timeout = 30

        # Run the download and parse operations in a thread pool
        # to avoid blocking the async event loop
        await asyncio.get_event_loop().run_in_executor(None, article.download)
        await asyncio.get_event_loop().run_in_executor(None, article.parse)

        if not article.text:
            raise Exception(f"No text content found in article: {url}")

        cached_articles[url] = article.text.strip()

        return cached_articles[url]

    except Exception as e:
        # Fallback to basic HTML scraping if newspaper3k fails
        try:
            return await _fallback_scrape(url)
        except Exception as fallback_error:
            raise Exception(
                f"Failed to scrape article from {url}. "
                f"Primary error: {str(e)}. "
                f"Fallback error: {str(fallback_error)}"
            )


async def _fallback_scrape(url: str) -> str:
    """
    Fallback scraping method using BeautifulSoup for basic text extraction
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status} error when fetching {url}")

            html = await response.text()

    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
        script.decompose()

    # Try to find main content areas commonly used by news sites
    content_selectors = [
        "article",
        ".article-content",
        ".article-body",
        ".entry-content",
        ".post-content",
        ".story-body",
        ".article-text",
        '[data-module="ArticleBody"]',
        ".RichTextStoryBody",
        ".InlineVideo-container ~ p",
    ]

    article_text = ""
    for selector in content_selectors:
        content = soup.select(selector)
        if content:
            article_text = " ".join([elem.get_text(strip=True) for elem in content])
            break

    # If no specific content area found, extract from all paragraphs
    if not article_text:
        paragraphs = soup.find_all("p")
        article_text = " ".join(
            [
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 50
            ]
        )

    if not article_text or len(article_text) < 100:
        raise Exception("Could not extract meaningful article content")

    return article_text.strip()
