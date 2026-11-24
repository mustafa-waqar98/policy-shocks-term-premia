"""Small download helper. Every fetcher saves the raw file so runs are reproducible."""

from pathlib import Path

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (research script; policy-shocks-term-premia)"}


def fetch(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
