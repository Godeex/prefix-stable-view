# Environment & data setup — I-01 reproduction

> 打包时**实际重跑了哪些脚本、哪些没重跑**（本机缺 numpy/scipy/torch）记录在同目录 `VERIFICATION.md`。
> Which scripts were actually re-run at packaging time — and which were not, because numpy/scipy/torch
> were absent — is recorded in `VERIFICATION.md` alongside this file.

Everything runs **locally and offline after the models and two datasets are present**. No cloud
API keys, no paid services, no new package installs beyond the four scientific libs in
`requirements.txt`. All LLM and embedding traffic goes to one local **Ollama** server over HTTP
(`http://127.0.0.1:11434`) using the Python standard library, so the scripts never import an LLM
SDK.

## 1. Python environment

The reported numbers were produced under **Python 3.10.20**. Any 3.10+ works.

```bash
# conda (the env used for the paper)
conda create -n i01 python=3.10 && conda activate i01
# OR venv
# python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt   # numpy, scipy, pandas, statsmodels
```

In every command below, `python` means the interpreter with those four packages installed. The
scripts are always invoked as `python x.py`, so the `#!/usr/bin/env python` shebang on line 1 is
never relied on, and no script records which interpreter produced a number.

## 2. Model server (Ollama) — REQUIRED for all `run_*.py` and `walking_skeleton.py`

Start a local Ollama server and pull the models. The cost/quality probes are decoder-only
transformers; two model families reproduce every headline, and a third (gemma2) probes
attention-design generality.

```bash
ollama serve                       # listens on 127.0.0.1:11434
# chat/generation models
ollama pull qwen3:14b              # primary cost-leg + quality model
ollama pull qwen2.5:14b            # 2nd family, reproduces the cost leg to within 0.1 token
ollama pull qwen3:8b               # walking-skeleton + hazard model
ollama pull qwen3.8:27b            # (optional) largest hazard-slice model
ollama pull llama3:latest          # (optional) independent family for hazard generality
ollama pull mistral:latest         # (optional) independent family for hazard generality
ollama pull gemma2:2b              # (optional) interleaved local/global attention probe
# embedding model (danger-slice prevalence scan + co/embedder.py)
ollama pull bge-m3                 # used for near-twin semantic similarity; degrades to lexical
                                   # overlap if unavailable (scripts print DEGRADED)
```

Hardware note: models in the paper run on consumer GPUs (RTX 4090 / L40 class, ~24 GB), one at a
time. The concurrency probe (`run_prefix_concurrency.py`) fires N sessions at ONE shared Ollama
slot; it does not need N GPUs. No distributed or multi-GPU setup is assumed.

Verify before a run:

```bash
curl -s http://127.0.0.1:11434/api/tags | grep -o qwen3:14b
```

## 3. Datasets — REQUIRED by the `gen_*.py` generators

Two corpora. Each generator reads its corpus path from an **environment variable** (`LB`, `DATA`,
or `MTU`); there is **no baked-in default** — if the variable is unset the script exits with a
message naming the variable to set. Point them wherever your local copies live; nothing in the
package assumes a particular machine's directory layout.

| env var | file / dir it should point at | used by | source |
|---|---|---|---|
| `LB` | `…/longbench/data/hotpotqa.jsonl` | `gen_prefix_cost.py`, `gen_prefix_bounded.py` | LongBench (hotpotqa split) |
| `DATA` | `…/longbench/data/` (dir holding `hotpotqa.jsonl`, `2wikimqa.jsonl`, `musique.jsonl`) | `gen_joint_sessions.py`, `gen_lb_danger.py` | LongBench |
| `MTU` | `…/mtu/M-S_eval.jsonl` | `gen_policy_trials.py`, `gen_danger_trials.py` | MTU-Bench (Multi-Turn-Understanding) eval split |

LongBench and MTU-Bench are open corpora; retrieve them through the Hugging Face **mirror** (the
author network cannot reach huggingface.co directly), then export the variables to wherever you put
them:

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
# LongBench (contains hotpotqa.jsonl under data/)
huggingface-cli download AIburp/LongBench --repo-type dataset --local-dir "$HOME/data/longbench"
# MTU-Bench evaluation set (M-S_eval.jsonl); adjust the repo id to the MTU release you cite in the paper
huggingface-cli download <mtu-bench-repo> --repo-type dataset --local-dir "$HOME/data/mtu"

# then tell the generators where they landed:
export LB="$HOME/data/longbench/data/hotpotqa.jsonl"
export DATA="$HOME/data/longbench/data"
export MTU="$HOME/data/mtu/M-S_eval.jsonl"
```

If you only want the **cost leg** (Sec. "Cost", the head result), you need **LongBench hotpotqa
only** — `gen_prefix_cost.py` / `gen_prefix_bounded.py` draw real record prose from it and the
recall slot is synthetic. MTU is only needed for the field/three-arm `policy` experiment.

> Do not ship the datasets in this folder; they are large and license-bound. Regenerate the trial
> `.jsonl` files with the `gen_*.py` scripts (Section "Run" in README).

## 4. Nonces (cache correctness)

`run_prefix_cost.py`, `run_prefix_bounded.py` and `run_prefix_concurrency.py` prepend a **per-run +
per-arm nonce** (env `PC_NONCE`, default = current timestamp) to every prompt. This is load-bearing:
llama.cpp's radix cache keys on the byte prefix, so without a fresh nonce a stale whole-prompt node
from a previous run would fake a cache hit for the *sliding* `length_only` window and destroy the
measurement. Leave `PC_NONCE` unset (it auto-fills) unless you deliberately resume a partially run
session.

## 5. Runtime pinning (content-addressed build)

The paper records the embedded llama.cpp build as *unknown* because `ollama -v` does not expose its
commit. That gap is closed by digest rather than by guessing: the exact binaries and model files
that produced every number are addressed below. Rebuilding or re-pulling anything whose digest
differs changes the tokenizer, the chat template, or the cache implementation, and the token
counts will move with them.

```
ollama version            0.33.3
/usr/bin/ollama                       sha256 7deaad14177b824d8fcff7136d4418c0111d2db35fa872a5071fa8db5b8db527
/usr/lib/ollama/libllama.so           sha256 af4d01b0a315516b61facc7ff9df13c65e3409162cf7bed3d5a48bb7fef83a0e
/usr/lib/ollama/libllama-server-impl.so sha256 47558fc8e3947942b6f5b942eb7a0d10744d32a80c2de515f09c0789e48c8e11
GPU / driver              3 x NVIDIA GeForce RTX 4090 + 1 x NVIDIA L40, driver 535.230.02
                          (two of the four GPUs are shared with other users' processes)
model digest (registry sha256)                          size bytes        family / params / quant
qwen3:14b        bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8  9276198565  qwen3   14.8B  Q4_K_M
qwen2.5:14b      7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6  8988124069  qwen2   14.8B  Q4_K_M
qwen3:8b         500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41  5225388164  qwen3    8.2B  Q4_K_M
qwen3.8:27b      22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643 17741872154  qwen35  27.3B  Q4_K_M  (imported from a local GGUF, parent tag qwen3.8:27b-q4_K_M; server-reported architecture qwen35, NOT qwen3; text path only)
llama3:latest    365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1  4661224676  llama    8.0B  Q4_0
mistral:latest   6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf  4372824384  llama    7.2B  Q4_K_M
gemma2:2b        8ccf136fdd5298f3ffe2d69862750ea7fb56555fa4d5b18c04e3fa4d82ee09d7  1629518495  gemma2   2.6B  Q4_0
bge-m3:latest    7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab  1157672605  bert   566.7M  F16
```

Re-produce them before a run and compare:

```bash
ollama -v; sha256sum /usr/bin/ollama /usr/lib/ollama/libllama.so
curl -s http://127.0.0.1:11434/api/tags | python -c "import json,sys; [print(m['name'], m['digest'], m['size']) for m in json.load(sys.stdin)['models']]"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Per-request settings the server applies: `temperature` as recorded per run, `num_ctx=8192`,
`stream=false`, `think=false`; `keep_alive` and the cache TTL are left at server defaults (the
client never sets them).

## 6. Token-compressor arm (external competitor, Sec. "External content rule") — OPTIONAL, needs `torch` + `transformers`

`compress_llmlingua2.py` builds the LLMLingua-2-style token-pruning control arms
(`comp_open50` / `comp_open25` / `comp_open20` / `comp_open12` / `comp_keep25` / `comp_keep20`) from
the cost-leg trials, and `run_prefix_comp.py` replays them against the same server. This is the one
part of the package that needs a second model, installed separately from Ollama:

```bash
pip install torch transformers          # the scorer is a 177M-param BERT token classifier
export HF_HUB_OFFLINE=1                 # if the snapshot is already cached (author network cannot
                                        # reach huggingface.co directly)
# OR, to fetch it the first time:
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

The model id is overridable with `CC_MODEL`; the pruning rates with `CC_RATES` (default
`0.50,0.25,0.20,0.12`). `CC_MAX_SUB` caps the scorer's context at 480 subword tokens (the release is a
512-position BERT; the script packs word-aligned chunks under that limit instead of silently
truncating). Scoring is deterministic — no sampling, one scoring pass per trial serves every rate,
and the rate is encoded in the arm name (`comp_open20` means "keep a $.20$ fraction of the window's
whitespace tokens"), which is what lets a rate be added to a leg that is already being served without
touching any earlier arm's bytes. `CC_KEEP` names the rates whose arms also get a
gold-protected twin (`comp_keep*`, the answer row force-kept). It takes about two minutes for all
552 trials on 1 CPU thread-set; `check_comp_controls.py`
re-scores a prefix of the trials in a temp directory and asserts the prompts come back
byte-identical.

The budget-matched rate is not guessed: `matched_budget.py` computes it from the run files, by
regressing server-measured prompt tokens on mean kept-word count over the arms already served in this
leg (the naive tokens-per-word proxy is printed alongside and rejected, because the pruner keeps the
token-dense words). That fit is therefore evaluated on **partial serving of the leg whose rate it
picks** — a disclosed post-hoc element of the design, recorded as Amendment 2 in
Sec.~"External content rule" of the paper and in `run_prefix_comp.py`'s pre-declared reading. Once the
leg is complete the script reports the fit's pick and the arm the leg actually served side by side;
the paper quotes the served arm, and the paired bootstrap on prompt tokens decides whether the match
holds. `CMP_NONCE` pins the per-run nonce; the $.20$ arms were served under the same
nonce as the arms added before them, so their prompts are byte-identical to what those arms saw.

Note on hardware: the authors' torch build (`2.12.0+cu130`) could not use their GPUs (the machine's
CUDA driver is older than the toolkit), so the shipped arm was scored **on CPU**. That does not
affect any number in the paper — the pruner is a separate model from the answerer, it only decides
which words survive, and the answer/cost measurements are all taken by Ollama.

## 7. Optional / not used

- **vLLM / SGLang**: deliberately NOT installed. The paper's cost numbers are a llama.cpp/Ollama
  in-process radix-cache **proxy**; a paged, continuously-batched TTFT/throughput headline is left
  as future work (see `results/prefix_concurrency_report.md` for what a shared-pool probe does and
  does not establish).
- **matplotlib / plotting**: paper figures are drawn programmatically in TikZ from the reported
  numbers; no plotting library is imported by these scripts.
- **transformers (opt-in, two probes)**: apart from the compressor arm above, no `gen_*/run_*/analyze_*`
  script imports it. The other exception is `verify_cache_measure.py`: parts A (proxy fidelity),
  B (token-prefix reuse) and C (smoke calibration) need a real HF tokenizer to check the server's
  own cached-token counters, taken from `PB_PROXY` (default `Qwen/Qwen2.5-VL-7B-Instruct`), so
  rebuilding them needs `pip install transformers` and network or mirror access to that repo. The
  report shipped in `results/cache_verify_report.json` was run with `PB_PARTS=CDE`; part C is
  therefore the one section in the package that a `requirements.txt`-only environment cannot
  reproduce, while D (eviction probe) and E (server facts) run on the four pinned packages
  (`PB_PARTS=DE`). Nothing on the confirmatory path — the trial files, the scored run files, or any
  statistic in the paper — depends on it.
