"""Download helper with a hard time limit per file and a few retries.

Some government sites throttle requests from cloud servers (like GitHub Actions)
by sending data very slowly. A plain timeout only limits the gap between chunks,
so a slow trickle could hang forever. Here every attempt has a total deadline.
"""

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("tpshock")

# Identify honestly as a script. Pretending to be a browser gets requests stalled by bot
# protection on some sites (FRED, for one), because the connection doesn't look like Chrome's.
HEADERS = {
    "User-Agent": "policy-shocks-term-premia/0.1 (research project; github.com/mustafa-waqar98)",
    "Accept": "*/*",
}


def fetch(url: str, dest: Path, deadline: int = 90, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        start = time.monotonic()
        try:
            with requests.get(url, headers=HEADERS, timeout=(15, 30), stream=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                        if time.monotonic() - start > deadline:
                            raise TimeoutError(f"download exceeded {deadline}s")
            tmp.replace(dest)
            return dest
        except (requests.RequestException, TimeoutError) as exc:
            last_err = exc
            log.warning("attempt %d/%d for %s failed: %s", attempt, retries, url, exc)
            time.sleep(3 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"could not download {url}: {last_err}")
