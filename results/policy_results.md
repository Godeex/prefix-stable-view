# M-A real-load comparison (qwen3:8b, temp=0) — E6 analysis

## matched-token check
- per-(base,K) char-length spread max = 1 (0 => exactly matched)
- mean prompt_eval_count by arm: length_only=185.0, m_a=189.8, worst=181.2

## contamination / fidelity by arm
- length_only n=90 contamination=0.000 (Wilson[0.000,0.041]) mean_nsim=0.400 mean_max_sim=0.458 gold_present=1.000
- m_a         n=90 contamination=0.000 (Wilson[0.000,0.041]) mean_nsim=0.317 mean_max_sim=0.335 gold_present=1.000
- worst       n=90 contamination=0.144 (Wilson[0.086,0.232]) mean_nsim=0.528 mean_max_sim=0.554 gold_present=1.000

## at-risk paired contrast (only groups where length_only kept a NEARER twin than m_a)
- all-at-risk: n=61 LO_contam_rate=0.000 MA_contam_rate=0.000 no discordant pairs
- high-at-risk(max_sim>0.45): n=36 LO_contam_rate=0.000 MA_contam_rate=0.000 no discordant pairs

## beta refit on NATURAL field values (logistic contamination ~ max_sim + K)
- beta0=-8.372 beta_max_sim=8.161 (p=0.00109) beta_K=0.552 (p=0.165)
- McFadden pseudo-R2=0.145  VIF(max_sim|K)=1.04
- reading: positive & significant beta_max_sim => semantic proximity predicts contamination on natural fields (surrogate revalidated off the artificial near/far contrast).

wrote configs/calibration_params.natural.json (fit_scope=stage6-natural, provisional)