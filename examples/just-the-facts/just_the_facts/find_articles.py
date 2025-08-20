import random

import feedparser

feeds = {
    "NBC News Top Stories": "http://feeds.nbcnews.com/feeds/topstories",
    "BBC News Top Stories": "https://feeds.bbci.co.uk/news/rss.xml",
    "CBS News Top Stories": "http://www.cbsnews.com/latest/rss/main",
    "Fox News Latest": "http://feeds.foxnews.com/foxnews/latest",
}

all_urls = []

for name, url in feeds.items():
    print(f"\n=== {name} ===")
    feed = feedparser.parse(url)

    for entry in feed.entries[:25]:
        print(entry.link)
        all_urls.append(entry.link)


# shuffle
random.shuffle(all_urls)

print(all_urls)
