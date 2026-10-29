# M-A conditional-benefit test on the DANGER slice (A)

danger prevalence (naive length_only retains near-twin, lo_max_sim>=t): {'0.45': 195, '0.5': 122, '0.52': 92, '0.55': 71, '0.58': 26, '0.6': 26} of 345 groups
threshold used=0.52; matched-token char spread reported by generator.

## qwen2.5:14b temp=0.0 seed=1234
### all danger groups (>=0.52)  (n_paired=91)
- contamination: length_only=0.000[0.000,0.041]  m_a=0.000[0.000,0.041]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=1.000  m_a=0.989
- one-sided (M-A better) signal: tie
### high-danger subset (>=0.55)  (n_paired=70)
- contamination: length_only=0.000[0.000,0.052]  m_a=0.000[0.000,0.052]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=1.000  m_a=0.986
- one-sided (M-A better) signal: tie

## qwen3:14b temp=0.0 seed=1234
### all danger groups (>=0.52)  (n_paired=91)
- contamination: length_only=0.000[0.000,0.041]  m_a=0.000[0.000,0.041]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=1.000  m_a=1.000
- one-sided (M-A better) signal: tie
### high-danger subset (>=0.55)  (n_paired=70)
- contamination: length_only=0.000[0.000,0.052]  m_a=0.000[0.000,0.052]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=1.000  m_a=1.000
- one-sided (M-A better) signal: tie

## qwen3:8b temp=0.0 seed=1234
### all danger groups (>=0.52)  (n_paired=91)
- contamination: length_only=0.000[0.000,0.041]  m_a=0.000[0.000,0.041]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=0.978  m_a=0.978
- one-sided (M-A better) signal: tie
### high-danger subset (>=0.55)  (n_paired=70)
- contamination: length_only=0.000[0.000,0.052]  m_a=0.000[0.000,0.052]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=0.971  m_a=0.971
- one-sided (M-A better) signal: tie

## qwen3:8b temp=0.7 seed=1234
### all danger groups (>=0.52)  (n_paired=91)
- contamination: length_only=0.000[0.000,0.041]  m_a=0.000[0.000,0.041]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=0.978  m_a=0.978
- one-sided (M-A better) signal: tie
### high-danger subset (>=0.55)  (n_paired=70)
- contamination: length_only=0.000[0.000,0.052]  m_a=0.000[0.000,0.052]
- McNemar discordant: b(LO dirty,MA clean)=0  c(MA dirty,LO clean)=0  exact p=1
- fidelity floor gold_present: LO=0.971  m_a=0.971
- one-sided (M-A better) signal: tie

## cross-config verdict
- M-A direction (b>c) in all configs: False
- per-config b>c (full danger slice): qwen2.5:14b temp=0.0 seed=1234: b=0/c=0 p=1; qwen3:14b temp=0.0 seed=1234: b=0/c=0 p=1; qwen3:8b temp=0.0 seed=1234: b=0/c=0 p=1; qwen3:8b temp=0.7 seed=1234: b=0/c=0 p=1
- at least one config individually significant (p<0.05): False
## diagnosis: why the danger slice shows zero contamination (cross-check vs (1))
- In (1) the 13 worst-arm contaminations clustered at exactly two near-duplicate field values
  (max_sim=0.527 x11, all K>=2; and 0.736 x3). Crucially **50 of 63 worst-arm trials with
  distractor_max_sim>=0.52 did NOT contaminate** at temp=0 -> high bge-m3 cosine is a WEAK proxy for
  actual confusability; only a handful of genuine near-duplicate string-pairs confuse the model.
- The danger-slice length_only arm reaches max_sim up to 0.715 (7 groups >=0.65), K skewed to 2-3,
  yet still **0 contamination in both arms across 3 families (qwen2.5:14b, qwen3:14b, qwen3:8b) and
  temp in {0.0, 0.7}** -> the natural single-field-retrieval arena almost never exposes the specific
  confusable near-duplicates that M-A's eviction is meant to drop.

## honest reading (NOT an M-B overturn; a narrowing of the M-A systems claim)
- M-B mechanism (semantic proximity -> contamination) stands: it was confirmed on constructed
  near-distractors, and revalidated directionally by the (1) logistic refit (beta_max_sim=8.16,
  p=0.001). This A test does NOT refute the mechanism.
- BUT M-A's operational benefit *over naive truncation* is **undemonstrable on realistic MTU-Bench
  field retrieval** at local 8-14B scale: the dangerous natural regime (co-present field that both
  survives naive truncation AND actually confuses the model) is rare enough that matched-pair
  McNemar yields b=c=0 everywhere. The fidelity floor holds (gold_present ~0.98-1.0 both arms).
- Consequence for the paper's story: single-field contamination on easy retrieval is the WRONG arena to
  show M-A's value. Distraction risk had teeth on the HARDER loads already probed (EX-2a entity
  roster ΔR²~0.5; EX-2b LongBench passage substitution ΔR²~0.11) — M-A should be evaluated there,
  or its value re-argued on the cost/prefix-stability/fidelity axes rather than contamination delta.
