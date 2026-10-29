# M-A vs naive truncation on the PASSAGE danger slice (①)

usable questions=300 K=4 trunc=500
naive positional truncation retains near-adversary passage (lo_max>=t): {'0.55': 74, '0.6': 47, '0.65': 23, '0.7': 14, '0.75': 9, '0.8': 3}
threshold used=0.6

## gemma2:2b temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.383[0.258,0.526]  m_a=0.340[0.222,0.483]
- McNemar: b(LO subst,MA clean)=9  c(MA subst,LO clean)=7  exact p=0.8036
- correct rate: LO=0.149  m_a=0.191
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.286[0.117,0.546]  m_a=0.429[0.214,0.674]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=3  exact p=0.625
- correct rate: LO=0.143  m_a=0.000
- M-A better signal: NO c>b

## gemma2:2b temp=0.6 seed=777
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.362[0.240,0.505]  m_a=0.340[0.222,0.483]
- McNemar: b(LO subst,MA clean)=9  c(MA subst,LO clean)=8  exact p=1
- correct rate: LO=0.128  m_a=0.213
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.286[0.117,0.546]  m_a=0.429[0.214,0.674]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=3  exact p=0.625
- correct rate: LO=0.071  m_a=0.000
- M-A better signal: NO c>b

## llama3:latest temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.213[0.120,0.349]  m_a=0.128[0.060,0.252]
- McNemar: b(LO subst,MA clean)=7  c(MA subst,LO clean)=3  exact p=0.3438
- correct rate: LO=0.234  m_a=0.362
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.143[0.040,0.399]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.143  m_a=0.286
- M-A better signal: NO c>b

## llama3:latest temp=0.6 seed=777
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.213[0.120,0.349]  m_a=0.128[0.060,0.252]
- McNemar: b(LO subst,MA clean)=7  c(MA subst,LO clean)=3  exact p=0.3438
- correct rate: LO=0.234  m_a=0.362
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.143[0.040,0.399]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.143  m_a=0.286
- M-A better signal: NO c>b

## mistral:latest temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.234[0.136,0.372]  m_a=0.085[0.034,0.199]
- McNemar: b(LO subst,MA clean)=10  c(MA subst,LO clean)=3  exact p=0.09229
- correct rate: LO=0.234  m_a=0.234
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.000[0.000,0.215]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=0  exact p=1
- correct rate: LO=0.143  m_a=0.214
- M-A better signal: YES b>c

## mistral:latest temp=0.6 seed=777
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.234[0.136,0.372]  m_a=0.085[0.034,0.199]
- McNemar: b(LO subst,MA clean)=9  c(MA subst,LO clean)=2  exact p=0.06543
- correct rate: LO=0.213  m_a=0.255
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.000[0.000,0.215]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=0  exact p=1
- correct rate: LO=0.071  m_a=0.286
- M-A better signal: YES b>c

## qwen2.5:14b temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.149[0.074,0.277]  m_a=0.021[0.004,0.111]
- McNemar: b(LO subst,MA clean)=6  c(MA subst,LO clean)=0  exact p=0.03125
- correct rate: LO=0.255  m_a=0.319
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.071[0.013,0.315]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=0  exact p=1
- correct rate: LO=0.143  m_a=0.429
- M-A better signal: tie

## qwen2.5:14b temp=0.6 seed=777
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.149[0.074,0.277]  m_a=0.064[0.022,0.172]
- McNemar: b(LO subst,MA clean)=6  c(MA subst,LO clean)=2  exact p=0.2891
- correct rate: LO=0.255  m_a=0.319
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.071[0.013,0.315]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=0  exact p=1
- correct rate: LO=0.143  m_a=0.429
- M-A better signal: tie

## qwen3.8:27b temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=44)
- substitution: length_only=0.136[0.064,0.267]  m_a=0.068[0.023,0.182]
- McNemar: b(LO subst,MA clean)=6  c(MA subst,LO clean)=3  exact p=0.5078
- correct rate: LO=0.386  m_a=0.477
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=13)
- substitution: length_only=0.077[0.014,0.333]  m_a=0.077[0.014,0.333]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.231  m_a=0.385
- M-A better signal: tie

## qwen3.8:27b temp=0.5 seed=1234
### danger slice (lo_max>=0.6) (n_paired=44)
- substitution: length_only=0.159[0.079,0.294]  m_a=0.068[0.023,0.182]
- McNemar: b(LO subst,MA clean)=7  c(MA subst,LO clean)=3  exact p=0.3438
- correct rate: LO=0.386  m_a=0.477
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=13)
- substitution: length_only=0.077[0.014,0.333]  m_a=0.077[0.014,0.333]
- McNemar: b(LO subst,MA clean)=1  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.231  m_a=0.385
- M-A better signal: tie

## qwen3:14b temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.234[0.136,0.372]  m_a=0.128[0.060,0.252]
- McNemar: b(LO subst,MA clean)=7  c(MA subst,LO clean)=2  exact p=0.1797
- correct rate: LO=0.277  m_a=0.319
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.143[0.040,0.399]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.143  m_a=0.357
- M-A better signal: NO c>b

## qwen3:14b temp=0.6 seed=777
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.234[0.136,0.372]  m_a=0.149[0.074,0.277]
- McNemar: b(LO subst,MA clean)=7  c(MA subst,LO clean)=3  exact p=0.3438
- correct rate: LO=0.277  m_a=0.340
- M-A better signal: YES b>c
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.071[0.013,0.315]  m_a=0.143[0.040,0.399]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.143  m_a=0.357
- M-A better signal: NO c>b

## qwen3:8b temp=0.0 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.170[0.089,0.301]  m_a=0.170[0.089,0.301]
- McNemar: b(LO subst,MA clean)=2  c(MA subst,LO clean)=2  exact p=1
- correct rate: LO=0.213  m_a=0.191
- M-A better signal: tie
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.143[0.040,0.399]  m_a=0.214[0.076,0.476]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.071  m_a=0.143
- M-A better signal: NO c>b

## qwen3:8b temp=0.7 seed=1234
### danger slice (lo_max>=0.6) (n_paired=47)
- substitution: length_only=0.170[0.089,0.301]  m_a=0.170[0.089,0.301]
- McNemar: b(LO subst,MA clean)=2  c(MA subst,LO clean)=2  exact p=1
- correct rate: LO=0.213  m_a=0.213
- M-A better signal: tie
### high-danger subset (lo_max>=0.70) (n_paired=14)
- substitution: length_only=0.143[0.040,0.399]  m_a=0.214[0.076,0.476]
- McNemar: b(LO subst,MA clean)=0  c(MA subst,LO clean)=1  exact p=1
- correct rate: LO=0.071  m_a=0.214
- M-A better signal: NO c>b

## cross-config verdict
- M-A direction (b>c) in all configs: False
- per-config (danger slice): gemma2:2b temp=0.0 seed=1234: b=9/c=7 p=0.804 n=47; gemma2:2b temp=0.6 seed=777: b=9/c=8 p=1 n=47; llama3:latest temp=0.0 seed=1234: b=7/c=3 p=0.344 n=47; llama3:latest temp=0.6 seed=777: b=7/c=3 p=0.344 n=47; mistral:latest temp=0.0 seed=1234: b=10/c=3 p=0.0923 n=47; mistral:latest temp=0.6 seed=777: b=9/c=2 p=0.0654 n=47; qwen2.5:14b temp=0.0 seed=1234: b=6/c=0 p=0.0312 n=47; qwen2.5:14b temp=0.6 seed=777: b=6/c=2 p=0.289 n=47; qwen3.8:27b temp=0.0 seed=1234: b=6/c=3 p=0.508 n=44; qwen3.8:27b temp=0.5 seed=1234: b=7/c=3 p=0.344 n=44; qwen3:14b temp=0.0 seed=1234: b=7/c=2 p=0.18 n=47; qwen3:14b temp=0.6 seed=777: b=7/c=3 p=0.344 n=47; qwen3:8b temp=0.0 seed=1234: b=2/c=2 p=1 n=47; qwen3:8b temp=0.7 seed=1234: b=2/c=2 p=1 n=47
- at least one config individually significant (p<0.05): True
## (a) CROSS-ARCHITECTURE probe — gemma2:2b (analyze_multiseed.py + pull)
Pulled gemma2:2b: different vendor AND a materially different attention design (interleaved
local/global sliding-window attention + qk soft-capping) vs the standard GQA of llama/qwen/mistral.
Honest ceiling set up front: no non-transformer model is pullable locally (mamba/SSM absent from
ollama), so this tests cross-FAMILY + cross-ATTENTION-design, NOT architecture-CLASS independence.
Danger slice THR>=0.55 (n=74), both seeds:
  gemma2 t0/s1234 : LO 0.383 -> MA 0.340, b=12/c=10 (near-null, sign b>c preserved)
  gemma2 t0.6/s777: LO 0.362 -> MA 0.340, b=13/c=12 (near-null)
Reading: gemma2 does NOT refute the effect (direction holds) but its OWN magnitude is capability-
limited — a 2B model substitutes from remaining distractors almost as often as it evicts, like the
qwen3:8b tie. Reinforces "benefit SIZE tracks capability, not architecture".
Pooled sensitivity: adding gemma2's 2 strata to the MH moves 8-strata p=6.35e-8 -> 10-strata p=1.49e-6
(T 29.25->23.16; the near-null gemma2 layers dilute chi2 but the claim SURVIVES a cross-arch model).
Cluster bootstrap unchanged (4 capable models): D=0.0997 CI[+0.037,+0.166] P(D<=0)=1.2e-3 SIG.
VERDICT (a): cross-attention/family generality SUPPORTED (direction replicates, pooled robust);
architecture-CLASS independence still UNTESTED (would need an encoder-decoder / SSM, not pullable).

---

## (a-REV) POOLING REVISION (declared a priori) — supersedes the pooled block above

SUPERSEDED, DO NOT QUOTE: the "Pooled sensitivity: 8-strata p=6.35e-8 -> 10-strata
p=1.49e-6 (T 29.25->23.16)" and "D=0.0997 CI[+0.037,+0.166] P(D<=0)=1.2e-3" lines in
section (a) above. They were written before the llama3 re-run landed, are NOT
reproducible from the current `lb_danger_scored_*.jsonl`, and the manuscript no longer
quotes them.  Re-run `analyze_pooled.py` (no argument = a-priori primary, `capable` =
subgroup) to regenerate; the fresh output is saved as `pooled_all.txt` /
`pooled_capable.txt`.

CURRENT, from the 14 served (model,temp) strata on the >=0.60 danger slice:

  PRIMARY (all 14 strata, no capability filter, declared a priori):
    naive pooled McNemar      B=94  C=43  p=1.568e-05      (anti-conservative)
    Mantel-Haenszel, 14 cells sum(b-c)=51 sum(b+c)=137 T=18.99 p=1.317e-05
    Mantel-Haenszel, 7 indep. temp0 models      T=10.88  p=9.718e-04
    QUESTION-cluster bootstrap, 47 questions / 652 paired rows
                              D=0.0782  95% CI [+0.0031,+0.1631]  P(D<=0)=0.0205

  SUBGROUP (5 capable checkpoints = 10 strata; criterion visible in the per-config
  table, so reported as subgroup, not primary):
    Mantel-Haenszel, 10 cells T=24.00 p=9.634e-07
    Mantel-Haenszel, 5 indep. temp0 models T=13.30 p=2.657e-04
    bootstrap, 464 paired rows D=0.1034 95% CI [+0.0261,+0.1922] P(D<=0)=0.004

  DIRECTION: M-A strictly below naive in 12/14 strata; the two qwen3:8b cells tie (2/2).
  PER-STRATA McNemar: only qwen2.5:14b@t0 clears 0.05 (b=6 c=0, p=0.031); mistral@t0.6
  is the closest miss (p=0.065); gemma2's two cells are 0.80 / 1.00.
  SESOI CHECK: the primary point estimate (0.0782) clears the declared 0.03 floor but
  its 95% CI does not (lower bound 0.0031) -> reported as a direction-consistent benefit
  of unpin-down size, NOT as a demonstrated SESOI-sized effect.
  CROSS-ARCH CHECK (gemma2 at the looser >=0.55 cutoff, n=74): t0 b/c=12/10,
  rates 0.338 -> 0.311; t0.6 b/c=13/12, rates 0.324 -> 0.311.  (The old lines above
  quoted 0.383->0.340 for this slice; those were the >=0.60 rates copied in error.)

VERDICT (a-REV): cross-attention/family generality still SUPPORTED on direction;
architecture-CLASS independence still UNTESTED.  The pooling claim no longer depends
on a filter chosen after seeing the effect.
