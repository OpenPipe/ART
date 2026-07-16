#!/usr/bin/env python3
"""List GitHub release assets without the REST release endpoint."""

from __future__ import annotations

import argparse
import json
import os
from urllib import request

_QUERY = """
query($owner: String!, $name: String!, $tag: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    release(tagName: $tag) {
      releaseAssets(first: 100, after: $cursor) {
        nodes { name downloadUrl }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def release_assets(repository: str, tag: str, token: str) -> list[dict[str, str]]:
    owner, name = repository.split("/", 1)
    cursor: str | None = None
    assets: list[dict[str, str]] = []
    while True:
        body = json.dumps(
            {
                "query": _QUERY,
                "variables": {
                    "owner": owner,
                    "name": name,
                    "tag": tag,
                    "cursor": cursor,
                },
            }
        ).encode()
        http_request = request.Request(
            "https://api.github.com/graphql",
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(http_request, timeout=30) as response:
            payload = json.load(response)
        if errors := payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL error: {errors}")
        release = payload["data"]["repository"]["release"]
        if release is None:
            return []
        page = release["releaseAssets"]
        assets.extend(
            {"name": node["name"], "url": node["downloadUrl"]} for node in page["nodes"]
        )
        if not page["pageInfo"]["hasNextPage"]:
            return assets
        cursor = page["pageInfo"]["endCursor"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    print(json.dumps({"assets": release_assets(args.repository, args.tag, token)}))


if __name__ == "__main__":
    main()
