**Paper:** *Maintaining LLM-Agent Conversation History as a Prefix-Stable Streaming Materialized View* 

**Supplementary Material for Paper:** *supplementary.pdf*

## 1. What is in the archive

| count | kind                                                 | note                                                                                                                                                                                                                    |
| ----- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 175   | files in the archive                                 | 130 at the top level, 8 in `co/`, 34 in `results/`, 2 in `configs/`, 1 in `tests/`                                                                                                                                      |
| 54    | `.jsonl`                                             | 39 `*scored*` run files (one row per model call, with the server's own cached-token / prompt-token / prefill counters) plus 15 trial and session definition files that record what was issued before any model answered |
| 60    | top-level Python scripts, plus `run_index_sk_leg.sh` | generate, gate, serve, analyse, and the table writers                                                                                                                                                                   |
| 8     | `co/`                                                | the minimal `ContextOptimizer` system the experiments drive (`admission`, `assembler`, `distraction_cost`, `embedder`, `eval_sink`, `history_log`, `summary_view`, `__init__`)                                          |
| 1     | `tests/test_probes.py`                               | the unit probes, run as `PYTHONHASHSEED=0 python tests/test_probes.py`                                                                                                                                                  |
| 34    | `results/`                                           | the prose and numeric captures the manuscript's tables and figures were copied from (11 `.txt`, 14 `.json`, 9 `.md`)                                                                                                    |
| 3     | `*.tex`                                              | `index_tables.tex`, `index_form.tex`, `comp_table.tex`: floats written by scripts, typed by nobody                                                                                                                      |
| 2     | `configs/`                                           | the frozen design settings (thresholds, arms, grids)                                                                                                                                                                    |

No dataset is shipped. LongBench and MTU-Bench material is fetched at run time (section 4).

## 2. One pass over the shipped logs

```bash
pip install -r requirements.txt      # numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, statsmodels 0.14.6
python headline_numbers.py           # prints every ratio, CI and p-value quoted in the paper,
                                     # and writes headline_numbers.json
```

`headline_numbers.py` reads only the `.jsonl` files in this archive. It needs the four pinned
libraries but no server, no model and no corpus: the model calls were made once, and every quoted
statistic is a function of the logs. Scripts divide by what they need:

| needs                                                     | scripts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| nothing beyond the standard library, no server, no corpus | the deterministic pre-run gates `check_joint_layout`, `check_layout_controls`, `check_position_controls`, `check_index_controls`, `check_index_sk_controls`, and `check_comp_controls` with `CC_RERUN=0`; `tests/test_probes.py` (run as `PYTHONHASHSEED=0 python tests/test_probes.py`); `sample_flow.py`; `fold_cost.py`; `power_mde.py`; the four trial builders that read only shipped definition files: `gen_index_trials`, `gen_index_cost_trials`, `gen_index_sk_trials`, `gen_position_trials` |
| the four pinned libraries                                 | `headline_numbers.py`, every `analyze_*.py`, `equivalence_tost.py`, `matched_budget.py`, `make_index_tables.py`, `make_comp_table.py`                                                                                                                                                                                                                                                                                                                                                                  |
| a local Ollama server                                     | every `run_*.py`, `walking_skeleton.py`, `probe_cache_field.py`, `verify_cache_measure.py`; and the four generators that call the embedder while building trial sets: `gen_joint_sessions.py`, `gen_lb_danger.py` (LongBench `DATA`), `gen_danger_trials.py`, `gen_policy_trials.py` (MTU-Bench `MTU`)                                                                                                                                                                                                 |
| corpus files, no server                                   | `gen_prefix_cost.py` and `gen_prefix_bounded.py`, which read LongBench (`LB`) and write trial files without model traffic                                                                                                                                                                                                                                                                                                                                                                              |

| `torch` + `transformers` (opt-in) | `compress_llmlingua2.py`; the determinism re-score at the end of `check_comp_controls.py` (default `CC_RERUN=25`, which invokes that script and exits 1 if it is missing, after the control checks have printed and passed); parts A to C of `verify_cache_measure.py` |

## 3. Manuscript location to script to run file

Body sections are numbered as in the submitted version; supplementary sections are named by title.
Table and figure references are given as the manuscript's own labels. The three script-written
floats (`comp_table.tex`, `index_tables.tex`, `index_form.tex`) ship in this archive and are
byte-identical to the copies the manuscript reads, so no cell of those tables was typed. The
supplementary's run-inventory section carries the same map with per-run sample sizes, and
`results/sample_flow.md` is written from the logs by `sample_flow.py`.

| manuscript                                                                                                                                                                                                                                  | study                                                                                                                  | generate, serve, analyse                                                                                                                                                                                                                                                                                                           | logs and capture                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sec. 5.3 measured maintenance cost, with Table `tab:foldcost` in the supplementary                                                                                                                                                          | fold accounting on the bounded arm                                                                                     | `run_prefix_bounded.py` then `fold_cost.py`                                                                                                                                                                                                                                                                                        | `prefix_bounded_scored_*.jsonl`, `results/prefix_bounded_report.md`                                                                                                                                                    |
| Sec. 6.1 setup, statistics, measurement validity                                                                                                                                                                                            | cache-field validity and counter trust                                                                                 | `probe_cache_field.py`, `verify_cache_measure.py`, `summarize_cache_verify.py`, `repeat_ms_variance.py`, `sample_flow.py`                                                                                                                                                                                                          | `ms_repeat_variance_scored.jsonl`, `results/cache_verify_report.json`, `results/ms_repeat_variance.json`, `results/sample_flow.md`                                                                                     |
| Sec. 6.1, Suppl. "Statistical procedure in full" / "SESOI anchoring"                                                                                                                                                                        | pooling rule, equivalence test, design precision                                                                       | `analyze_pooled.py`, `equivalence_tost.py`, `power_mde.py`                                                                                                                                                                                                                                                                         | `results/pooled_all.txt`, `results/pooled_size.txt`, `results/pooled_subgroup.txt`                                                                                                                                     |
| Sec. 6.2, Table `tab:cost` and Fig. `fig:cost`                                                                                                                                                                                              | cost leg, 3 arms x 552 turns, two models                                                                               | `gen_prefix_cost.py`, `run_prefix_cost.py`, `analyze_prefix_cost.py`                                                                                                                                                                                                                                                               | `prefix_cost_scored_*.jsonl`, `results/prefix_cost_report.md`                                                                                                                                                          |
| Sec. 6.3, Table `tab:comp` (the float `comp_table.tex`)                                                                                                                                                                                     | external content rule at matched prompt budget (LLMLingua-2 class pruning)                                             | `compress_llmlingua2.py`, `check_comp_controls.py`, `run_prefix_comp.py`, `analyze_prefix_comp.py`, `matched_budget.py`, `make_comp_table.py`                                                                                                                                                                                      | `prefix_comp_scored_*.jsonl`, `comp_table.tex`, `results/comp_analysis_summary.json`, `results/matched_budget.json`                                                                                                    |
| Sec. 6.4, Table `tab:position`                                                                                                                                                                                                              | gold-position stress inside the head, 720 cells x 2 architectures                                                      | `gen_position_trials.py`, `check_position_controls.py`, `run_position_stress.py`, `analyze_position.py`                                                                                                                                                                                                                            | `position_scored_*.jsonl`, `results/position_analysis.txt`, `results/position_analysis.json`                                                                                                                           |
| Sec. 6.4, Table `tab:twoxtwo` and Fig. `fig:interaction`                                                                                                                                                                                    | joint versus decoupled eviction, 4 configs x 408 turns                                                                 | `gen_joint_sessions.py`, `run_joint_2x2.py`, `analyze_joint.py`, `analyze_2x2_correct.py`                                                                                                                                                                                                                                          | `joint_scored_*.jsonl`, `results/joint_2x2_report.md`                                                                                                                                                                  |
| Sec. 6.4, Fig. `fig:generality`                                                                                                                                                                                                             | tail stress under forced commit, its unforced original, and the second-architecture replication over the same sessions | `gen_joint_sessions.py` for the base sessions (the tail session file ships as `joint_sessions_tail.jsonl`), then `run_joint_2x2.py` with `RJ_MODE=tail`, `RJ_FORCE=1`, `RJ_MODEL`, `RJ_OUT`, and `analyze_joint.py`                                                                                                                | `joint_scored_qwen3_14b_tail.jsonl`, `joint_scored_qwen3_14b_tailf.jsonl`, `joint_scored_qwen3_14b_tailf_epoch2.jsonl`, `joint_scored_llama3_latest_tailf.jsonl`, `results/joint_2x2_report.md`                        |
| Sec. 6.5                                                                                                                                                                                                                                    | finite head budget with periodic folding; 8-way contention on one shared KV pool                                       | `gen_prefix_bounded.py`, `run_prefix_bounded.py`, `analyze_prefix_bounded.py`; `run_prefix_concurrency.py`                                                                                                                                                                                                                         | `prefix_bounded_scored_*.jsonl`, `prefix_concurrency_scored_*.jsonl`, `results/prefix_concurrency_report.md`                                                                                                           |
| Sec. 6.6, with the floats `tab:index-fid`, `tab:index-cost`, `tab:index-sk` (`index_tables.tex`) and `tab:index-form` (`index_form.tex`) in the supplementary, plus Suppl. "Head-internal position stress and the price of the index block" | rewritable `INDEX` view: fidelity leg, cost leg, shared-key re-servation                                               | `gen_index_trials.py` / `check_index_controls.py` / `run_index_stress.py` / `analyze_index.py`; `gen_index_cost_trials.py` / `run_index_cost.py` / `analyze_index_cost.py`; `gen_index_sk_trials.py` / `check_index_sk_controls.py` / `run_index_sk_leg.sh` / `analyze_index_sk.py`; `make_index_tables.py`, `diag_answer_form.py` | `index_scored_*.jsonl`, `index_cost_scored_*.jsonl`, `index_sk_scored_*.jsonl`, `index_tables.tex`, `index_form.tex`, `results/index_analysis.txt`, `results/index_cost_analysis.txt`, `results/index_sk_analysis.txt` |
| Sec. 6.7, Fig. `fig:forest`, with Table `tab:danger` in the supplementary                                                                                                                                                                   | passage-level danger on real multi-document QA                                                                         | `gen_lb_danger.py`, `run_lb_danger.py`, `analyze_lb_danger.py`, `analyze_pooled.py`, `analyze_capability.py`                                                                                                                                                                                                                       | `lb_danger_scored_*.jsonl`, `results/lb_danger_report.md`, `results/pooled_all.txt`, `results/capability_analysis.txt`                                                                                                 |
| Suppl. "Additional measured detail"                                                                                                                                                                                                         | the single-field null contrast and the three-arm field hazard run behind it                                            | `gen_danger_trials.py`, `run_danger_infer.py`, `analyze_danger.py`; `gen_policy_trials.py`, `run_policy_infer.py`, `analyze_policy.py`                                                                                                                                                                                             | `danger_scored_*.jsonl`, `policy_scored_*.jsonl`, `results/danger_report.md`, `results/policy_results.md`                                                                                                              |

| Suppl. "Layout-only factorial at fixed content" | labels versus prefix stability at fixed content | `run_joint_layout.py`, `check_layout_controls.py`, `analyze_layout.py` | `joint_layout_scored_*.jsonl`, `results/joint_layout_report.md`, `results/layout_analysis.json` |
| Suppl. "Capability anchor: the reading we dropped" | external capability anchor, and the reading that was withdrawn | `analyze_capability.py` | `results/capability_analysis.txt` |
| Suppl. "Content-rule leg: design, pre-declared predictions, amendments" | the two disclosed amendments on the competitor leg | `run_prefix_comp.py` (predictions are written in the file), `analyze_prefix_comp.py`, `matched_budget.py` | `results/comp_analysis_summary.json`, `results/matched_budget.json` |
| Suppl. "Run inventory, sample flow and maintenance ledger" | how many calls, which files, what was excluded | `sample_flow.py`, `headline_numbers.py` | `results/sample_flow.md`, `results/sample_flow.json`, `headline_numbers.json` |

## 4. Environment of record

`ENVIRONMENT.md` is the long form; the facts that decide whether a re-run reproduces a number:

- Python 3.10.20 with the four pinned libraries in `requirements.txt`. `numpy` is not optional for
  the analysis layer: `make_index_tables.py` and `make_comp_table.py` exit with
  `ModuleNotFoundError` without it.
- All model traffic goes to one local Ollama server at `http://127.0.0.1:11434` over the standard
  library `urllib`; no SDK is imported, so no client-version drift is possible. The runtime is
  pinned by digest rather than by a llama.cpp build string the server does not report:
  `ollama version 0.33.3`, with sha256 for `/usr/bin/ollama`, `/usr/lib/ollama/libllama.so` and
  `/usr/lib/ollama/libllama-server-impl.so` in `ENVIRONMENT.md` section 5.
- Serving was single-GPU (RTX 4090 / L40 class, about 24 GB) with one model resident at a time. The
  concurrency probe fires N sessions into the one shared Ollama slot; it is a contention probe, not
  a throughput benchmark.
- Corpus locations come from the environment variables `LB` (LongBench), `DATA` and `MTU`
  (MTU-Bench `M-S_eval.jsonl`) and have **no defaults**: unset, a script names the variable it
  needs. Material was fetched through `HF_ENDPOINT=https://hf-mirror.com` because the author
  network cannot reach `huggingface.co`. The cost leg needs only LongBench `hotpotqa`.
- The cost, bounded, index and competitor legs serve at `temperature=0`, `seed=1234`,
  `num_predict=40` and `num_ctx=8192`. The danger strata that use a non-zero temperature name both
  the temperature and the seed in their file names, so a mix-up is visible in the file list.
- `PC_NONCE` is load-bearing. The server's radix cache keys on the byte prefix of the prompt, so a
  re-run that reuses a nonce can hit the previous run's cached nodes and turn a sliding window's
  miss into a fake hit. Each run script takes a fresh nonce on its own.
- `bge-m3` supplies the embedding used by the hazard and near-twin definitions. Without that model
  the embedding call degrades to a lexical-overlap score and prints `DEGRADED`; a degraded pass is
  not comparable to the published rows.

## 5. Call audit

34,134 confirmatory model calls are recorded in 38 scored run files, plus a 276-call diagnostic
replication of 23 published cost turns in `ms_repeat_variance_scored.jsonl`: 34,410 calls across 39
scored files, 7 checkpoints over 5 reported architectures. `results/sample_flow.md` lists, per run,
the designed cells, the issued calls and the exclusions, and it is regenerated from the logs by
`sample_flow.py`, so the arithmetic quoted in the manuscript is checkable without a server.

## 6. What this package does not carry

- No datasets, no model weights, no server logs beyond the per-call counters in the run files.
- End-to-end serving throughput and tail latency under a continuously batched scheduler. The cost
  claims are in recomputed tokens; the prefill milliseconds are reported as agreement with that
  currency, on a host where other users' processes share two of the four GPUs.
- Figures. The manuscript's figures are drawn programmatically in TikZ from the reported numbers,
  so no plotting library is imported here and no figure file needs regenerating to check a number.
- A claim about any review process. Licence and citation fields are left as placeholders in this
  submission build on purpose.

---



# 论文复现代码包 · Agent 对话历史的前缀稳定流式物化视图维持

---

## 这是什么 / What this establishes（三条主张 / three claims）

| 主张 Claim                      | 结论（一句话）                                                                                                                                                                                                                                                                                                                                                                                                                                  | 证据文件 Evidence                                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **C2 · Cost**                 | 在**信息保真持平**下，前缀稳定维持比朴素长度截断每轮**重算少约 6.3×、prefill 快约 5×**；等长 pinned 对照复原 97% 的差距 → 收益来自**前缀失效**而非"少发 token"                                                                                                                                                                                                                                                                                                                                | `results/prefix_cost_report.md`                                                                                      |
| **C2b · Budget**              | 给 head 设**有限预算并周期性折叠**后，head 大小持平、仍保住约 **5.8×** 缩减，代价仅 +24 tok/轮折叠surcharge → 结论不依赖"无界 head"                                                                                                                                                                                                                                                                                                                                             | `results/prefix_bounded_report.md`                                                                                   |
| **C2c · Load**                | 8 路并发争用同一共享 KV 池时，三臂重算**次序与幅度不变**（naive−stable 配对差与单流相差 <1.5%）→ 该优势是**按内容键**的缓存性质，非独占访问假象                                                                                                                                                                                                                                                                                                                                                | `results/prefix_concurrency_report.md`                                                                               |
| **C3/C4 · Hazard & fidelity** | 朴素截断在保留**近孪生候选值**时易被污染（logistic 邻近项系数显著）；前缀感知维持把 gold 放进带标签 STATE 块从而抑制替换                                                                                                                                                                                                                                                                                                                                                                | `results/danger_report.md`, `results/lb_danger_report.md`                                                            |
| **C5 · I-03（null）**           | 在前缀稳定之上再叠"语义价值折叠"联合驱逐打分，在 matched budget 下**无额外收益**（两种机制互为替代品），如实报负                                                                                                                                                                                                                                                                                                                                                                      | `results/joint_2x2_report.md`, `results/policy_results.md`                                                           |
| **C6 · Position**             | gold 行在 head 内的**位置**本身就是保真轴：共享 key 压力下 front→back 正确率 $0.933\!\rightarrow\!0.008$（$H{=}24$，$\Delta-0.925$，$p{=}8{\times}10^{-34}$），llama3 同向；据此界定前缀稳定维持**能**把哪条信息放在哪儿                                                                                                                                                                                                                                                                   | `results/position_analysis.txt`, `results/sample_flow.md`                                                            |
| **C7 · Index（mixed）**         | 第二张**可重写**视图 `INDEX`（`key→value` 活绑定）把 head 内位置梯度从 $-0.375$ 压到 $-0.158$（DiD $+0.217$，$p{=}4.3{\times}10^{-6}$），代价是每行 7.40 个重算 token（CI $[7.19,7.62]$）；预声明判读 **P2 失败**（残余梯度仍显著），如实报 mixed。**共享-key hazard 复现**：把同一 `INDEX` 重架到共享-key `STATE`（distractor 亦键 `TARGET`，stale 值同键竞争）时块恢复崩塌——index−plain 正确率 $+0.479/+0.662$（qwen3/llama3，各对照 $p\le3.1{\times}10^{-12}/\le1{\times}10^{-21}$），且 stale index 把替换率抬到 raw state 之上，故收益非 re-key 假象 | `results/index_analysis.txt`, `results/index_cost_analysis.txt`, `results/index_sk_analysis.txt`, `index_tables.tex` |


---

## 目录结构 / Folder layout

```
code/
├── README.md              本文件（中英） this file (bilingual)
├── supplementary.pdf       Supplementary Material for Paper
├── requirements.txt       仅科学栈 4 包（无 torch/vllm/ollama-sdk）; only 4 scientific libs
├── ENVIRONMENT.md         运行环境：Ollama + 模型 tag + LongBench/MTU 数据 + HF 镜像│
├── co/                    M-A 系统最小实现（被实验驱动） minimal ContextOptimizer the experiments exercise
│   ├── history_log.py         append-only 历史日志（保真底座）
│   ├── summary_view.py        前缀稳定摘要视图（序列化→KV 字节前缀稳定）
│   ├── embedder.py            ollama bge-m3 嵌入；不可用降级词面 overlap
│   ├── distraction_cost.py    CBO 代价项 σ(β0+β1·nsim+β2·K)
│   ├── admission.py           联合打分 utility−前缀深度−distraction；VERBATIM/FOLD/EVICT
│   ├── assembler.py           拼 prompt：append-only STATE 头 + RECENT 尾；chat() 走 ollama
│   ├── eval_sink.py           污染/替换判定 RoundMetric
│   └── __init__.py            包标记（`from co import ...`）
├── configs/               β 先验（calibration_params*.json，标 PROVISIONAL）
├── tests/test_probes.py   3 条失败探针 + 单调性 property（离线、plain assert）
├── walking_skeleton.py    端到端合成会话跑全链（连通性证据）
│
├── probe_cache_field.py   [成本] 探针0：ollama 是否暴露可用前缀缓存信号
│   # —— 成本腿 Cost leg (C2) ——
├── gen_prefix_cost.py  run_prefix_cost.py  analyze_prefix_cost.py
│   # —— 有界折叠 Bounded-fold (C2b) ——
├── gen_prefix_bounded.py  run_prefix_bounded.py  analyze_prefix_bounded.py
│   # —— 并发 Concurrency (C2c) ——
├── run_prefix_concurrency.py
│   # —— 2×2 joint（I-03 负结果 C5） ——
├── gen_joint_sessions.py  check_joint_layout.py  run_joint_2x2.py  analyze_joint.py
│   # —— layout-only 2×2（标签 × 前缀稳定对照） ——
├── run_joint_layout.py  analyze_layout.py  check_layout_controls.py
├── analyze_2x2_correct.py     # 把 correct 拆成 id/金标内容/干扰内容/other 四 disjoint 类
│   # —— 危险切片/字段 Hazard（C3/C4） ——
├── gen_danger_trials.py  run_danger_infer.py  analyze_danger.py
├── gen_lb_danger.py  run_lb_danger.py  analyze_lb_danger.py
├── analyze_pooled.py  analyze_capability.py  analyze_multiseed.py  analyze_sweep.py
├── power_mde.py  equivalence_tost.py   # 设计精度（80% 功效 MDE / 所需 n）与 TOST 等价检验
│   #   analyze_pooled.py 无参数 = a-priori 主分析（全部 14 strata）；`size` = 只看参数量的
│   #   敏感性池（12 strata，论文引用的敏感性数）；`subgroup` = 已撤下的结局筛子组（provenance）
│   #   analyze_capability.py = 外部能力锚（参数量级）+ Cochran Q/I² + 精确 Spearman + 同模型双温度溯源
│   # —— gold 位置压力 Position stress（C5 scope） ——
├── gen_position_trials.py  check_position_controls.py  run_position_stress.py  analyze_position.py
│   # —— 可重写索引视图 INDEX（C7）：保真腿 + 成本腿 ——
├── gen_index_trials.py  check_index_controls.py  run_index_stress.py  analyze_index.py
├── make_index_tables.py         # 直接生成论文 \input 的两张浮动体（index_tables.tex / index_form.tex）
├── gen_index_cost_trials.py  run_index_cost.py  analyze_index_cost.py
│   # —— 共享-key 重服务（C7 的 hazard 复现）：把当前绑定的 INDEX 重新架到共享-key STATE 上 ——
├── gen_index_sk_trials.py  check_index_sk_controls.py  run_index_sk_leg.sh  analyze_index_sk.py
├── diag_answer_form.py          # POST-HOC：把非正确答复按答题形态拆开（exploratory，不支撑任何判读）
│   # —— 外部内容规则对照臂（C8）：需要 torch+transformers，见 ENVIRONMENT.md sec.6 ——
├── compress_llmlingua2.py  check_comp_controls.py  run_prefix_comp.py
├── analyze_prefix_comp.py  make_comp_table.py   # 后者生成 comp_table.tex（论文 \input 的那份由 --tex-out 刷新）
├── matched_budget.py            # 从运行日志算"matched prompt budget"该取哪个 keep-rate（事后披露项）
│   # —— 测量效度 / 样本流 / 折叠账本 / 延迟复现 ——
├── probe_cache_field.py  verify_cache_measure.py  summarize_cache_verify.py
├── sample_flow.py  fold_cost.py  repeat_ms_variance.py
│   # —— 三臂 MTU 字段 Policy（C5 支撑） ——
├── gen_policy_trials.py  run_policy_infer.py  analyze_policy.py
│   # —— 一键重算论文所有数字 ——
├── headline_numbers.py        # 从 *_scored_*.jsonl 重生成正文引用的每个比值/CI/p
│   # —— 顶层"活输出"：脚本把结果写到自身目录（HERE=脚本所在目录），所以
│   #   comp_analysis_summary.json / comp_controls_report.json / comp_table_source.json /
│   #   headline_numbers.json / matched_budget.json / position_analysis.json /
│   #   sample_flow.{md,json} / index_tables.tex / index_form.tex / comp_table.tex
│   #   既是**运行落点**也是发布件；`results/` 下同名者 = 论文定稿时的**发布快照**。
│   #   打包时逐字节比对：8 个有 results/ 对应件的顶层文件全部一致（见 VERIFICATION.md）。
│   # —— 运行日志（run files）：39 个 *_scored_*.jsonl，一行一次调用 ——
│   #   34,134 call 在 38 个文件里承载论文的确认性分析；第 39 个
│   #   （ms_repeat_variance_scored.jsonl，276 call）是延迟复现诊断，计入审计但不承载任何主张。
│   #   两个数由 sample_flow.py 从文件本身算出（results/sample_flow.md 的 Global audit 段）。
│
└── results/               报告（论文表格/图的来源） authoritative reports
    ├── prefix_cost_report.md  prefix_bounded_report.md  prefix_concurrency_report.md
    ├── joint_2x2_report.md  joint_layout_report.md  danger_report.md
    ├── lb_danger_report.md  policy_results.md  sample_flow.md  position_analysis.txt
    ├── pooled_all.txt (主分析)  pooled_size.txt (size-only 敏感性)
    │   pooled_subgroup.txt (已撤下, provenance)  subgroup_analysis.txt
    │   capability_analysis.txt (能力锚检验)  headline_numbers.{py,json,txt}
    ├── ms_repeat_variance.json  cache_verify_report.json  cache_verify_events.json
    ├── danger_prevalence.json  lb_danger_prevalence.json
    ├── index_analysis.txt (索引保真腿)  index_cost_analysis.txt (索引成本腿)
    │   index_sk_analysis.txt (共享-key 重服务腿)
    │   index_tables_gen.txt (make_index_tables.py 的 stdout，含来源行审计)
    ├── comp_analysis_summary.json (两条腿各自的 F/B/R 与 C8-P1..P4 判读)
    │   comp_controls_report.json (跑前离线门)  compression_report.json (pruner 逐 trial 的存活/预算计量)
    │   matched_budget.json (keep-rate 拟合与"服务的那一档"对照)  comp_table_source.json (论文表的每个数)
    └── layout_analysis.json  position_analysis.json  sample_flow.json
```

每个 `gen_*.py` 生成试验 `.jsonl`（与脚本同目录）→ `run_*.py` 打本地 Ollama 推理并落
`*_scored_*.jsonl` → `analyze_*.py` 出统计。三者用 `HERE=dirname(__file__)` 自洽，可从任意工作目录运行。

Each `gen_*.py` writes its trials `.jsonl` (next to the script) → `run_*.py` replays it against
local Ollama and appends `*_scored_*.jsonl` → `analyze_*.py` produces the statistics. All three
resolve paths from `HERE=dirname(__file__)`, so they run from any working directory.

---

## 快速开始 / Quick start

```bash
cd code
pip install -r requirements.txt
# 先按 ENVIRONMENT.md 起 Ollama，并 export 语料位置：LB / DATA / MTU（生成脚本没有默认路径，
# 未设置会直接报错点名要哪个变量；run_*/analyze_* 不需要语料，只读包内 trials）
# Bring up Ollama per ENVIRONMENT.md and export the corpus locations LB / DATA / MTU: the
# gen_* scripts have no default path and exit naming the variable they need. run_*/analyze_*
# need no corpus -- they read the shipped trials.

# 1) 离线自检（不触模型）offline self-check, no model
python tests/test_probes.py

# 2) 成本腿（论文头条 C2）cost leg            需 export LB=…/hotpotqa.jsonl
python gen_prefix_cost.py            # NSESSIONS/NTURNS/WINDOW/REC_SENTS 可调
python run_prefix_cost.py            # 需 Ollama UP；PC_MODEL 默认 qwen3:14b
python analyze_prefix_cost.py

# 3) 有界折叠（C2b）bounded-fold              需 export LB=…/hotpotqa.jsonl
python gen_prefix_bounded.py
python run_prefix_bounded.py
python analyze_prefix_bounded.py

# 4) 并发探针（C2c）concurrency
python run_prefix_concurrency.py --parallel 8 --nsample 8
```

---

## 逐实验运行 / Per-experiment run

所有 `run_*.py` 都通过环境变量控制；缺省值即可复现论文所用的默认档。下面列出最常调的旋钮。
`gen_*.py` 另需语料环境变量 `LB` / `DATA` / `MTU`（**没有默认路径**，未设置即报错点名所需变量，
见 `ENVIRONMENT.md` §3）；`run_*.py` 与 `analyze_*.py` 不需要语料，它们只读包内的 trials。
Every `run_*.py` is controlled by environment variables; the defaults reproduce the paper's
configuration. The `gen_*.py` scripts additionally require `LB` / `DATA` / `MTU` (**no default
path**; unset means the script exits naming the variable), while `run_*.py` and `analyze_*.py`
need no corpus and read the shipped trials. Key knobs:

### 成本腿 Cost leg — `gen_prefix_cost.py → run_prefix_cost.py → analyze_prefix_cost.py`

- 生成 env：`LB`(必需) `NSESSIONS`(20) `NTURNS`(30) `WINDOW`(8) `REC_SENTS`(6)
- 包内随附的 `prefix_cost_trials.jsonl`（552 行 / 24 session）对应 `NSESSIONS=24`，即下面那条命令；
  用默认 `NSESSIONS=20` 得到 460 行，实测这 460 行恰是 552 行文件的前 460 行（逐字节前缀）
- 推理 env：`PC_MODEL`(qwen3:14b) `PC_TEMP`(0.0) `PC_NUMCTX`(8192) `PC_NONCE`(自动)
- 三臂：`length_only`（朴素滑窗）/ `prefix_stable`（M-A append-only 头）/ `pinned_raw`（等长稳定对照）
- 主指标：`recompute = prompt_eval_count − prompt_eval_cached_count`，佐证 `prompt_eval_duration`

```bash
NSESSIONS=24 NTURNS=30 python gen_prefix_cost.py     # 需 export LB=…/hotpotqa.jsonl
PC_MODEL=qwen3:14b  python run_prefix_cost.py
python analyze_prefix_cost.py
# 跨族复现 reproduce on 2nd family:
PC_MODEL=qwen2.5:14b python run_prefix_cost.py
```

### 有界折叠 Bounded-fold — `gen_prefix_bounded.py → run_prefix_bounded.py → analyze_prefix_bounded.py`

- 生成 env：`LB`(必需) `NSESSIONS`(12) `NTURNS`(40) `WINDOW`(8) `REC_SENTS`(6) `H_MAX`(10) `FOLD_KEEP`(4)
- `H_MAX` = 头最多容纳的摘要行数；溢出时把最旧行折成一条 summary（改写头部字节→该轮缓存失效→重算尖峰=sawtooth）
- 新增臂 `bounded_stable` 与 `length_only`、无界 `prefix_stable` 在同一轮、同一 gold 下并列

```bash
NSESSIONS=12 NTURNS=40 H_MAX=10 FOLD_KEEP=4 python gen_prefix_bounded.py
PC_MODEL=qwen3:14b python run_prefix_bounded.py
python analyze_prefix_bounded.py     # 打印 append vs fold 的 sawtooth 与三个 bootstrap 对比
```

### 并发探针 Concurrency — `run_prefix_concurrency.py`（复用成本腿的 trials）

```bash
python run_prefix_concurrency.py --parallel 8 --nsample 8
# 报告每臂并发下的 mean recompute + wall/p95 延迟 + 吞吐；确认重算次序不变
```

### 2×2 joint（负结果）— `gen_joint_sessions.py → check_joint_layout.py → run_joint_2x2.py → analyze_joint.py`

`check_joint_layout.py` 是**离线门**（不打模型）：先断言 4 个配置真的把 DISTRACTION × PREFIXDEPTH
两轴分开，再花推理。`run_joint_2x2.py` env：`RJ_MODEL` `RJ_MODE` `RJ_SESS` `RJ_NONCE` 等。

```bash
python gen_joint_sessions.py          # 需 export DATA=…/longbench/data
python check_joint_layout.py          # deterministic gate, exit!=0 时不要继续
python run_joint_2x2.py
python analyze_joint.py               # 主效应 + 交互 + joint-vs-更优单因子 Pareto 判决
```

### 危险切片 Hazard（C3/C4）— `gen_danger_trials.py` / `gen_lb_danger.py` 及各自 run/analyze

需要 LongBench（`gen_lb_danger.py` 走真实多文档 QA）。字段 MTU 三臂在 `gen_policy_trials.py`
（需 MTU `M-S_eval.jsonl`）。`run_danger_infer.py` / `run_lb_danger.py` / `run_policy_infer.py` 打模型，
`analyze_*` 出流行率、配对显著性、logistic + FDR、多 seed/THR 稳健性。**具体数字见
`results/danger_report.md` / `lb_danger_report.md` / `policy_results.md`。**

### 池化检验 Pooled tests — `analyze_pooled.py`（池化规则 a priori 化）

池化集合**先验声明为「全部被服务的 (model, temp) stratum」，不加任何能力/显著性筛选**。
旧稿的「≥7B 且 temp-0 出现 b>c」子组用了**结局**当判据，机械地抬高效应，
因此该子组被**撤下（superseded）**，只作 provenance 保留；替代它的是**只看参数量的敏感性池**
（所有 ≥7B 的 checkpoint，无结局筛 = 除 gemma2:2b 外的 6 checkpoint × 2 温度 = 12 strata）。
论文引用的敏感性数字来自 `size` 档：

```bash
python analyze_pooled.py            > results/pooled_all.txt      # 主分析：14 strata, T=18.99
python analyze_pooled.py size       > results/pooled_size.txt     # size-only 敏感性：12 strata, T=22.15
python analyze_pooled.py subgroup   > results/pooled_subgroup.txt # 已撤下的结局筛子组：10 strata（provenance）
```

每档打印：所用 mode 与 stratum 集、每 stratum 的 b/c、(1) 朴素池化 McNemar（标 anti-conservative）、
(2) Mantel–Haenszel χ²₁ over 全部 cells、(2b) over 相互独立的 temp-0 模型 strata、
(3) 以 question 为簇的 bootstrap 风险差 D + 95% CI + P(D≤0)、(4) 设计精度（per-question SD、SE、
80% 功效的 MDE、把下界抬到 0.03 之上所需题数）。
mode 解析出问题时脚本会 **assert 失败**（例如 `size` 档里若仍出现 gemma2 的 cell 就直接报错），
避免「标签是敏感性池、数其实是全池」这类静默串档。

### 设计精度 Design precision — `power_mde.py`

```bash
python power_mde.py     # 从 lb_danger_scored_*.jsonl 算簇内 SD → SE → 80% 功效 MDE → 所需 n
```

### gold 位置压力 Position stress（C5 scope）— `gen_position_trials.py → run_position_stress.py → analyze_position.py`

`check_position_controls.py` 是**离线门**（不打模型）：断言同一 session、同一 head 长度下三个槽位
（first/middle/last）发出的 STATE 行**多重集相同、字符长度相同**、tail 与问题**逐字节相同**（重排不改
这些量），gold 恰好落在 $\{0, H/2, H-1\}$、竞争值仍在 head 内且绝对位置最多移一行、gold 值在每格
prompt 里都存在（掉分是读不到、不是没给），且渲染模板与论文已发表的 stable/labelled 臂一致。

```bash
python gen_position_trials.py
python check_position_controls.py   # deterministic gate, exit!=0 时不要继续
python run_position_stress.py       # 两架构各 720 call（PS_MODEL 切换）
python analyze_position.py          > results/position_analysis.txt
```

门里还打印论文 Sec. VI-C 引用到的三个规模事实：同一 (session, $H$) 组内三个槽位之间的 prompt
**字符长度差最大 0**（240 组全查），$H{=}6$ 平均 1,915 字符 / 529 token（最长 2,770 / 713），
$H{=}24$ 平均 4,794 字符 / 1,271 token（最长 6,447 / 1,650）。同一组数也由 `analyze_position.py`
和 `headline_numbers.py` 打印，正文那三句话里的每个数都能从随包脚本重出。

### 可重写索引视图 INDEX（C7）— 保真腿 + 成本腿

第二张视图：在 append-only 的 `STATE` 头之后插一段**可整块重写**的 `INDEX`（每条活绑定一行
`key -> value`），用来检验「把当前绑定单独摊平」能不能拆掉头内位置梯度，以及它值多少重算 token。

保真腿（`index` 三臂 × 与位置压力同一 session 池 / 同一 H×槽位网格）：`check_index_controls.py`
是**离线门**，断言 `INDEX` 块是臂间唯一差异、head 仍是逐字节前缀、索引行由上下文里的记录导出、
危险值与 gold 位置保持不变、内容仍是已发表的那套构造。

```bash
python gen_index_trials.py            # 3 arms x 720 cells = 2,160 trials（IX_HS/VTAIL 可调）
python check_index_controls.py        # deterministic gate, exit!=0 时不要继续
python run_index_stress.py            # IX_MODEL 切换：qwen3:14b 与 llama3:latest 各 2,160 = 4,320 call
python analyze_index.py          > results/index_analysis.txt
python make_index_tables.py      > results/index_tables_gen.txt   # 顺带写出论文 \input 的两张浮动体
python diag_answer_form.py --trials index_trials.jsonl --scored index_scored_qwen3_14b.jsonl
                                    # POST-HOC：非正确答复里到底是"换了记录"还是"换了答题形态"
```

成本腿（同一条流上并置 5 个渲染，量 `INDEX` 块的**整块重写**代价）：

```bash
python gen_index_cost_trials.py       # 5 arms x 460 turns = 2,300 trials
python run_index_cost.py             # IC_MODEL/IC_TEMP/IC_NUMCTX/IC_NONCE
python analyze_index_cost.py    > results/index_cost_analysis.txt
```

实测（qwen3:14b, temp 0, 460 paired turns / 20 sessions，五臂 accuracy 全 1.000 → matched-quality
前提成立）：mean recompute `prefix_stable` **250.8** → `index_I2` **278.5** / `index_I8` **342.6** /
`index_I24` **446.6** tok/turn，配对簇 bootstrap 溢价 **+27.7 [27.1,28.2] / +91.8 [89.5,93.9] /
+195.7 [190.3,200.7]**；把溢价对行数 $I$ 回归（1,380 obs，簇在 20 个 session 上）得斜率
**7.40 tok/行 CI [7.19,7.62], p=3.6e-24**，即 $R_{\text{index}}(t)=\Theta(I\cdot\Delta_{\text{idx}})+\Theta(V)$。
`length_only` 同流 1,566.8 tok/turn，三个索引档仍比朴素截断省 3.5–5.6×。
诚实边界：这条流的 key 每轮都是新的（`OBS_t`），没有任何绑定被改写，且斜率本身说明块是**从第一个
字节重新 prefill**，所以 7.40 tok/行是**整块重写**的价格（单行更新的代价上界），论文 T7 明写这一点。

共享-key 重服务腿（把 `INDEX` 架到真正会失效的 hazard 上）：

```bash
python gen_index_sk_trials.py        # 2 arms x 720 cells = 1,440 trials 定义（distractor 亦键 TARGET）
python check_index_sk_controls.py    # deterministic 门：plain 行与已发表 C7 cell 逐字节一致
bash run_index_sk_leg.sh             # 复用 run_index_stress.py，qwen3:14b + llama3:latest 各 1,440 = 2,880 call
python analyze_index_sk.py      > results/index_sk_analysis.txt
```

`index`（TARGET→gold，fresh）与 `index_nok`（TARGET→stale）两臂在此腿内新跑，`plain` 直接复用已发表
C7 网格的同名 cell（逐字节 STATE），故 scored 文件各 1,440 行、审计腿 2,880 call。实测：把当前绑定的
`INDEX` 重新架到共享-key `STATE`（gold 与 distractor 同键 `TARGET`，stale 值真正同键竞争）时，块**恢复**
了 raw state 上的崩塌——index−plain 正确率 qwen3 崩塌四槽均值 $+0.479$、llama3 $+0.662$（每对照
$p\le3.1\times10^{-12}/\le1\times10^{-21}$），stale index 命名竞争者把替换率抬到 raw state 之上
（$+0.217/+0.590$），故 C7 的收益不是 re-key 假象。诚实边界：qwen3 的 `index` 臂非承诺答复占比可达 $0.525$
（主结论落在替换消除上），llama3 的 within-index 位置梯度仍在（衰减不消失），且这是两行块、无 key 被重绑，
溢价仍是整块重写代价上界。

### 外部内容规则对照臂 Competitor arm — `compress_llmlingua2.py → run_prefix_comp.py → analyze_prefix_comp.py`

「rolling summary 不是同一件事」这个论证要成立，roster 里必须**真跑一个外部内容规则**作对照：
LLMLingua-2 类 token 压缩，在 **matched prompt budget** 下报 F/B/R 三坐标。这条腿就是它。
pruner 自己实现（不依赖 `llmlingua` 包）：`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
（177M，ungated）给历史区的每个 token 打 keep 概率，按固定比例取 top-k；剪枝在**原始字节的空白词粒度**
上做，所以每个压缩 prompt 都能被验证是它压缩对象的**子序列**（不重写、不换 surface）。

```bash
export HF_HUB_OFFLINE=1          # 模型已在本地 cache；网络不通时用 HF_ENDPOINT=https://hf-mirror.com 先拉
python compress_llmlingua2.py    # CC_MODEL/CC_RATES/CC_KEEP；552 trials x 6 臂 -> prefix_comp_trials.jsonl
python check_comp_controls.py    # 离线门：子序列性/探针尾字节一致、gold 存活重算、预算单调、
                                 # 每臂逐字节前缀预测器、随机重跑逐字节一致
python matched_budget.py         # 从已服务的 run 文件算"贴住 view 预算"的 keep-rate（见下 AMENDMENT 2）
python run_prefix_comp.py        # CMP_MODEL/CMP_TEMP/CMP_NONCE；7 臂 x 552 = 3,864 call（含同轮 length_only 对照）
python analyze_prefix_comp.py    # 目录里每个 prefix_comp_scored_*.jsonl 都单独一遍，绝不把两个模型 pool 在一起；
                                 # CP_FILE/CP_PUB 只决定哪条腿被标成 [primary leg]（另一条标 [replication]），
                                 # 两条腿的数都由同一次运行打印并合并写进 comp_analysis_summary.json
python make_comp_table.py        # 生成 comp_table.tex（写在脚本自己旁边，论文用 --tex-out 指过去刷新；
                                 # 腿没跑完/臂数不齐会拒绝出表）
CMP_MODEL=qwen2.5:14b python run_prefix_comp.py    # 第二个模型的复现（与成本腿一致的做法）
```

臂的定义：`comp_open50/25/20/12` = 纯 pruner 在 keep-rate .50/.25/.20/.12，**gold 记录行不加保护**——它在不在
压缩结果里是被**测量**的性质（`survives_*` 逐条计数），不假设；`comp_keep25/20` = 同样的 .25/.20 但把 gold 行
强行留下（benefit of the doubt，让"保真掉"不能被"你答案都删了"一句话挡掉）。rate 直接写进臂名
（`comp_open20` 即"保留窗口空白词的 $.20$"），所以**往一条正在服务的腿里加一档不会改动已有臂的字节**：
所有臂来自同一条确定性 rank，每一档只是它的一个前缀（这条不变量由 `check_comp_controls.py` 复核）。

判读在 `run_prefix_comp.py` 里**开跑前写死**（C8-P1 BLOAT / C8-P2 FIDELITY / C8-P3 RECOMPUTE /
C8-P4 PARETO），`analyze_prefix_comp.py` 原样打印再按实测数逐条判。两次**公开的事后修正（AMENDMENT）**：
(1) C8-P3 原写法拿"绝对 recompute"和朴素窗口比，而剪枝后的窗口本来就小，绝对值机械地更低——这个对比被 B
混淆；修正在约 17%（92–115/552 每臂）打完时写进文件、剩余调用之前，原文一字未删，机制检验改成 **R/B 占比**
与 **matched-budget** 对比。(2) 首批臂只**括住**了 view 的预算（$.25$ 在上、$.12$ 远在下），而要报的对比是
matched，于是在 $506$–$529/552$ 打完、$.20$ 两臂开跑**之前**加档；rate 由 `matched_budget.py` 用"已服务臂的
server 实测 B 对保留词数回归"选出——**这个拟合用的正是它要选 rate 的那条腿的部分服务**，论文与本文件都如实
标注。腿跑完后该脚本并排报两件事：完整腿上拟合的取值（$.21$，未服务）与实际服务的臂（$.20$），论文引用的
是**服务的那一档**，而"是否真 matched"由 $B$ 的配对 bootstrap 区间裁定，不由拟合裁定。

### 延迟复现 Latency replication — `repeat_ms_variance.py`

共享 GPU 机上把成本腿已发表的 23 个 turn × 2 会话 × 2 臂重打 3 遍（每次换 nonce，276 call），
报**重复之间**的漂移：naive 臂 mean prefill 幅度 9.1%、stable 臂 3.9%，而 recompute 只动 0.2 / 0.0
token —— 论文的延迟区间因此被明确定性为「跨会话变异」而非测量噪声，成本主张一律用 token。

```bash
python repeat_ms_variance.py        # 需本地 Ollama；写 ms_repeat_variance_scored.jsonl + results/ms_repeat_variance.json
```

### 能力锚点检验 Capability anchor — `analyze_capability.py`

「收益幅度随能力增长」这句话必须用**外部锚**检验，不能靠看完表再挑子集。
同一模型两个温度**边际率相同**的行要交代是不是复制。一条命令出四组数：

```bash
python analyze_capability.py > results/capability_analysis.txt
```

1. 每个模型（只取 temp-0 stratum，七个模型相互独立）的风险差 D + question 簇 bootstrap CI；
2. Cochran Q / I²：七个 D 的散布是否超出抽样噪声（实测 Q=5.54, df=6, p=0.48, I²=0%）；
3. 对**外部锚**（tag/模型卡报的参数量级 2B–27B，离线可核）的 Spearman ρ + 全 7! 排列精确
   双侧 p + leave-one-model-out 区间（实测 ρ=+0.11, p=0.82, 区间 [−0.29,+0.50]）——
   因此论文不写「capability-tracking」，改写「方向一致、幅度未被规模区分」；
4. 同模型两温度的溯源：共同题集、seed（t0=1234 / 高温=777 或同 1234）、逐条 outcome 相同的
   对数、prompt token 相同而采样答案不同 → 证明是两次独立调用而非复制行。

```bash
python analyze_2x2_correct.py       # 把 2×2 每格答案分为 id / 金标内容 / 干扰内容 / other
                                    # 四 disjoint 类，并给 c01→c11、c10→c11 的配对 McNemar
                                    # （解释 correct 列 0.696→0.512 的下降是答题格式迁移）
```

---

## 预期结果（成本三腿，已核对）/ Expected results (the three cost legs, verified)

**Cost leg** (qwen3:14b, temp 0, 552 paired turns / 24 sessions), mean per-turn recompute tokens:

| arm                  | acc   | mean recompute | prefill ms |
| -------------------- | ----- | -------------- | ---------- |
| length_only          | 1.000 | 1557           | 1056       |
| prefix_stable (M-A)  | 1.000 | 249            | 205        |
| pinned_raw (control) | 1.000 | 285            | 230        |

LO − M-A = **1308.6** CI [1280.1, 1331.0]；LO − pinned = 1272.1；≈ **6.3×** reduction；qwen2.5:14b 复现到 0.1 token 以内。

**Bounded-fold** (qwen3:14b, 396 turns / 12 sessions, 48 fold events): sawtooth append **247** /
fold spike **456**, amortized mean **272**; head stays flat (454 tok) vs unbounded 498 tok.
LO − bounded = **1292.6** CI [1271.1, 1315.5] → **5.8×** under budget (unbounded 6.3×); fold
surcharge bounded − unbounded = **+23.8** CI [23.3, 24.4].

**Concurrency** (parallel 8, 8 sessions, 120 calls): recompute length_only **1583** ≫ pinned
**506** > prefix_stable **293**; ordering and magnitude preserved (naive−stable gap within 1.5% of
single-stream). Wall-clock stays flat across arms (~8 s) because Ollama serializes requests — so
this shows **recompute-token invariance under a shared pool, NOT a throughput/TTFT win** (see
`results/prefix_concurrency_report.md`).

---

## 复现性 / Reproducibility

- 所有成本腿 `seed=1234`、`temperature=0`、`num_ctx=8192`；每臂是各自**升序轮次独立缓存流**，前缀按真实多轮对话方式预热。
- **nonce 是承重的**：见 ENVIRONMENT.md §4。跨 run 复跑请让它自动取新值，否则会命中上一轮残留的整段 prompt 缓存节点，把滑窗的"缓存命中"伪造成真命中、毁掉测量。
- 试验 `.jsonl` 由生成器**确定性**产出（固定 `random.seed`），可复跑。**每 prompt 的事件日志随包发**：顶层 39 个
  `*_scored_*.jsonl`（含逐 call 的 prompt/completion eval 计数；其中 38 个承载论文的确认性分析，第 39 个是
  `ms_repeat_variance_scored.jsonl` 延迟复现诊断）+ `trials_*.jsonl` / `joint_sessions*.jsonl` / `position_{trials,sessions}.jsonl` /
  `index_{trials,cost_trials}.jsonl` / `danger_trials.jsonl` / `prefix_*_trials.jsonl`，即论文与 `results/*` 每个数字的原始来源；`analyze_*.py` 直接读它们，
  不连模型也能复算全部统计量。**第三方语料本身不随包发**（LongBench 等大且受原许可约束），包内只保留我方导出切片。
- **派生切片的许可义务（务必读）**：随包的 `danger_*` / `lb_*` / `policy_*` 等 `.jsonl` 是从公开基准
  （LongBench 的 hotpotqa / 2wikimqa / musique，以及 MTU）**派生**的切片，其中嵌有这些数据集的**逐字短段文本**。
  它们不因"被我们切进会话"而变成我们的原创内容：再分发本包时必须保留对这些数据集的引用与各自的
  attribution / licence 条款（各上游 license 以官方仓库为准，本包不代为声明、也不臆造其条款）。
- **Licence obligation for the derived slices**: the shipped `danger_*` / `lb_*` / `policy_*` `.jsonl` files
  are derived from public benchmarks (LongBench hotpotqa / 2wikimqa / musique, and MTU) and embed short
  **verbatim passages** from those datasets. Cutting them into our sessions does not make that text ours:
  redistribute this package with citations to those datasets and keep their attribution and licence terms
  intact. Each upstream licence is whatever its official repository states — this package neither re-declares
  nor invents those terms.
- **本包内已实测重跑**：在 `code/` 里跑 `sample_flow.py`、`headline_numbers.py`、`analyze_position.py`、
  `analyze_index.py`、`analyze_index_cost.py`、`analyze_index_sk.py`、`make_index_tables.py`，stdout 与 `results/*` 的对应文件、
  以及 `position_analysis.json` / `headline_numbers.json` 逐字节相同；`make_index_tables.py` 生成的
  `index_tables.tex` / `index_form.tex` 与论文 `\input` 的两份文件逐字节相同（这两张表的每个数都由脚本写入，
  表头一行写着 `% generated ... -- do not edit a number here`）。**打包这趟**（一台没装
  numpy/scipy/torch 的机器）只重跑了其中的 stdlib 子集，逐条 exit code、复现出来的数与**未重跑清单**
  写在 `VERIFICATION.md`。外部内容规则腿的离线三件
  （`analyze_prefix_comp.py`、`matched_budget.py`、`make_comp_table.py`）同样在本包内重跑过：
  `results/comp_analysis_summary.json` / `results/matched_budget.json` 逐字节相同，生成的 `comp_table.tex`
  与论文 `\input` 的那份逐字节相同；换 `CP_FILE` 只改哪条腿被标为 primary，两条腿的数一字不动。
- All cost runs use `seed=1234`, `temperature=0`, `num_ctx=8192`; each arm is its own
  ascending-turn cache stream so the prefix warms like a real conversation. Trial `.jsonl` files are
  generated deterministically (fixed seed). **The per-prompt event logs are shipped** (top-level
  `*_scored_*.jsonl` — 38 files carrying the paper's confirmatory analyses, plus the 276-call
  latency-replication diagnostic — together with the `trials_*` / `*_sessions*` / `danger_trials` /
  `prefix_*_trials` / `index_*_trials` files):
  every number in the paper and in `results/*` is recomputed from them by `analyze_*.py` without touching
  a model. Third-party corpora (LongBench etc.) are not shipped (large + license-bound); only our own
  derived slices are included.
- **Re-run checked inside this package**: `sample_flow.py`, `headline_numbers.py`, `analyze_position.py`,
  `analyze_index.py`, `analyze_index_cost.py`, `analyze_index_sk.py` and `make_index_tables.py` reproduce the corresponding
  `results/*` files byte for byte (twice in a row, so the match is not a timestamp accident), and the two
  floats `make_index_tables.py` emits are byte-identical to the `index_tables.tex` / `index_form.tex`
  that the manuscript `\input`s. Those tables carry a header line saying they are generated, and no
  number in them is typed by hand.
- **What the packaging pass itself re-ran** is recorded separately in `VERIFICATION.md`: that pass ran on
  a machine without numpy/scipy/pandas/statsmodels/torch, so it re-confirmed the stdlib subset (the four
  control gates, `tests/test_probes.py`, `sample_flow.py`, `fold_cost.py`, `power_mde.py`), found every
  top-level file that has a `results/` counterpart byte-identical, and lists the scripts it could **not**
  re-run there. The full-stack reruns claimed in the bullet above were done in the pinned environment.
