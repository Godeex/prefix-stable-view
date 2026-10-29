"""mod-context-assembler-ollama: assemble prompt = cacheable prefix ++ verbatim tail.

kv_blackbox (ADR-0002): prefix stability is expressed purely as byte layout of the
prompt. The summary view text is emitted FIRST and only ever grows append-only, so
`view.prefix_text()` is a byte prefix of every assembled prompt -> the serving engine
can reuse the KV prefix without any fine-grained cache API.
"""
import json
import urllib.request

OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"


class ContextAssembler:
    def __init__(self, model="stub", endpoint=OLLAMA_CHAT, temperature=0.0, num_predict=64):
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.num_predict = num_predict

    def assemble(self, view, verbatim, question="Report the requested value exactly."):
        prefix = view.prefix_text()
        tail = "\n".join(f"- {u.text}" for u in verbatim)
        # cacheable head = append-only STATE, emitted FIRST so it is a leading byte
        # prefix of every prompt (ADR-0002); instructions live in the changing tail.
        head = (prefix + "\n") if prefix else ""
        return (f"{head}RECENT TOOL RESULTS:\n{tail}\n"
                f"Using STATE and RECENT above, answer.\nQUESTION: {question}\nANSWER:")

    def chat(self, prompt):
        """Real ollama call for the end-to-end skeleton. Returns (text, prompt_eval_count)."""
        body = {"model": self.model, "stream": False, "think": False,
                "options": {"temperature": self.temperature, "num_predict": self.num_predict},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        txt = (data.get("message") or {}).get("content", "")
        return txt.strip(), int(data.get("prompt_eval_count") or 0)
