import json
from pathlib import Path

class SimpleHTTPCache:
    def __init__(self, path: str = ".cache/headers.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.db = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            self.db = {}

    def get_headers(self, url: str):
        return self.db.get(url, {})

    def set_headers(self, url: str, headers: dict):
        keep = {k: v for k, v in headers.items() if k in ("ETag","Last-Modified")}
        if keep:
            self.db[url] = keep
            self.path.write_text(json.dumps(self.db, indent=2), encoding='utf-8')
