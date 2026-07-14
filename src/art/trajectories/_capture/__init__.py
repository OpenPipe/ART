from __future__ import annotations

from threading import Lock

_installed = False
_lock = Lock()


def install() -> None:
    global _installed
    if _installed:
        return
    with _lock:
        if _installed:
            return
        from .aiohttp import install as install_aiohttp
        from .httpx import install as install_httpx
        from .requests import install as install_requests

        install_httpx()
        install_aiohttp()
        install_requests()
        _installed = True
