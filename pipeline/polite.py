# -*- coding: utf-8 -*-
import time, random
from urllib.parse import urlparse
from urllib import robotparser

def can_fetch(url: str, user_agent: str = "Mozilla/5.0") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt is missing or unreadable, be conservative but allow
        return True

def polite_sleep(base: float = 1.0, jitter: float = 0.5):
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))
