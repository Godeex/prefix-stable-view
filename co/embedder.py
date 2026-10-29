"""ollama bge-m3 embedder adapter with on-disk cache (reuses pilot cache convention)."""
import json
import math
import os
import urllib.request

OLLAMA_EMBED = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "bge-m3:latest"


class Embedder:
    def __init__(self, cache_path=None, model=EMBED_MODEL, endpoint=OLLAMA_EMBED):
        self.model = model
        self.endpoint = endpoint
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self.hits = 0
        self.misses = 0

    def embed(self, text):
        t = str(text)
        if t in self.cache:
            self.hits += 1
            return self.cache[t]
        self.misses += 1
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"model": self.model, "input": t}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            v = json.loads(r.read())["embeddings"][0]
        self.cache[t] = v
        return v

    def flush(self):
        if self.cache_path:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)

    @staticmethod
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-9)
