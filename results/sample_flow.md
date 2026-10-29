# Sample flow of every study (generated from the run files, not typed)

All runs pin `num_ctx=8192`, `temperature=0`, `seed=1234`, `num_predict<=40`, and a fresh per-arm nonce. `err` counts calls that returned no token counts.

`generated` is records per session x sessions; `designed` is what the pre-declared turn
window keeps from that pool, fixed before any inference and identical across arms.

## Prefix cost (3 arms)

*exclusion rule:* turns WINDOW-1..NTURNS-1 (window full), set by the generator before any inference

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_only | 720 | 552 | 552 | 0 | 552 | 7--29 | 1575 | 1850 | 22.6% |
| pinned_raw | 720 | 552 | 552 | 0 | 552 | 7--29 | 1584 | 1786 | 21.8% |
| prefix_stable | 720 | 552 | 552 | 0 | 552 | 7--29 | 445 | 720 | 8.8% |

Total analysed 1656, total excluded 0. Largest prompt any arm reached: 1850 tokens (22.6% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Prefix cost, 2nd model

*exclusion rule:* same designed cells as the primary cost run

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_only | 720 | 552 | 552 | 0 | 552 | 7--29 | 1589 | 1864 | 22.8% |
| pinned_raw | 720 | 552 | 552 | 0 | 552 | 7--29 | 1598 | 1800 | 22.0% |
| prefix_stable | 720 | 552 | 552 | 0 | 552 | 7--29 | 459 | 734 | 9.0% |

Total analysed 1656, total excluded 0. Largest prompt any arm reached: 1864 tokens (22.8% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Joint 2x2 (4 configs)

*exclusion rule:* turns first-hazard-reveal..last (distractor in context), set before inference

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c00 | 480 | 408 | 408 | 0 | 408 | 3--19 | 566 | 1057 | 12.9% |
| c01 | 480 | 408 | 408 | 0 | 408 | 3--19 | 574 | 1066 | 13.0% |
| c10 | 480 | 408 | 408 | 0 | 408 | 3--19 | 565 | 1055 | 12.9% |
| c11 | 480 | 408 | 408 | 0 | 408 | 3--19 | 573 | 1064 | 13.0% |

Total analysed 1632, total excluded 0. Largest prompt any arm reached: 1066 tokens (13.0% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Joint 2x2 tail stress

*exclusion rule:* last 5 turns (both near-twins in the volatile tail), same rule for all four configs

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c00 | 480 | 120 | 120 | 0 | 120 | 15--19 | 839 | 1078 | 13.2% |
| c01 | 480 | 120 | 120 | 0 | 120 | 15--19 | 848 | 1087 | 13.3% |
| c10 | 480 | 120 | 120 | 0 | 120 | 15--19 | 838 | 1076 | 13.1% |
| c11 | 480 | 120 | 120 | 0 | 120 | 15--19 | 847 | 1085 | 13.2% |

Total analysed 480, total excluded 0. Largest prompt any arm reached: 1087 tokens (13.3% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Joint 2x2 tail stress, 2nd family

*exclusion rule:* same window as the qwen3 tail run; every row's model field verified llama3:latest

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c00 | 480 | 120 | 120 | 0 | 120 | 15--19 | 788 | 987 | 12.0% |
| c01 | 480 | 120 | 120 | 0 | 120 | 15--19 | 797 | 996 | 12.2% |
| c10 | 480 | 120 | 120 | 0 | 120 | 15--19 | 789 | 989 | 12.1% |
| c11 | 480 | 120 | 120 | 0 | 120 | 15--19 | 798 | 998 | 12.2% |

Total analysed 480, total excluded 0. Largest prompt any arm reached: 998 tokens (12.2% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Layout 2x2 (4 cells)

*exclusion rule:* same hazard-window turns as the joint 2x2, identical rule for all four cells

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0l0 | 480 | 408 | 408 | 0 | 408 | 3--19 | 567 | 1058 | 12.9% |
| s0l1 | 480 | 408 | 408 | 0 | 408 | 3--19 | 575 | 1067 | 13.0% |
| s1l0 | 480 | 408 | 408 | 0 | 408 | 3--19 | 563 | 1055 | 12.9% |
| s1l1 | 480 | 408 | 408 | 0 | 408 | 3--19 | 575 | 1067 | 13.0% |

Total analysed 1632, total excluded 0. Largest prompt any arm reached: 1067 tokens (13.0% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Bounded head (3 arms)

*exclusion rule:* turns WINDOW-1..NTURNS-1, identical rule for all three arms

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded_stable | 480 | 396 | 396 | 0 | 396 | 7--39 | 454 | 705 | 8.6% |
| length_only | 480 | 396 | 396 | 0 | 396 | 7--39 | 1589 | 1810 | 22.1% |
| prefix_stable | 480 | 396 | 396 | 0 | 396 | 7--39 | 498 | 744 | 9.1% |

Total analysed 1188, total excluded 0. Largest prompt any arm reached: 1810 tokens (22.1% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Gold-position stress, qwen3 (3 slots)

*exclusion rule:* every designed slot (first/middle/last x H in {6,24}) issued once; the grid is 120 sessions x 6 cells = 720 trials, 240 per slot arm; the row's model field is verified qwen3:14b, no post-hoc exclusion

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | 240 | 240 | 240 | 0 | 240 | single-turn | 900 | 1650 | 20.1% |
| last | 240 | 240 | 240 | 0 | 240 | single-turn | 900 | 1650 | 20.1% |
| middle | 240 | 240 | 240 | 0 | 240 | single-turn | 900 | 1650 | 20.1% |

Total analysed 720, total excluded 0. Largest prompt any arm reached: 1650 tokens (20.1% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Gold-position stress, llama3 (3 slots)

*exclusion rule:* same slots as the qwen3 position run, served a second time to llama3:latest (1,440 calls over both architectures); every row's model field verified llama3:latest

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | 240 | 240 | 240 | 0 | 240 | single-turn | 842 | 1567 | 19.1% |
| last | 240 | 240 | 240 | 0 | 240 | single-turn | 842 | 1567 | 19.1% |
| middle | 240 | 240 | 240 | 0 | 240 | single-turn | 842 | 1567 | 19.1% |

Total analysed 720, total excluded 0. Largest prompt any arm reached: 1567 tokens (19.1% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Rewritable-index fidelity, qwen3 (3 arms)

*exclusion rule:* every designed cell (H in {6,24} x first/middle/last) issued once per arm over the frozen gold-position session pool; the grid holds 3 arms x 720 cells = 2,160 trials and the row's own arm column splits them; the row's model field is verified qwen3:14b, no post-hoc exclusion

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index | 720 | 720 | 720 | 0 | 720 | single-turn | 926 | 1677 | 20.5% |
| index_nok | 720 | 720 | 720 | 0 | 720 | single-turn | 920 | 1670 | 20.4% |
| plain | 720 | 720 | 720 | 0 | 720 | single-turn | 900 | 1650 | 20.1% |

Total analysed 2160, total excluded 0. Largest prompt any arm reached: 1677 tokens (20.5% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Rewritable-index fidelity, llama3 (3 arms)

*exclusion rule:* same grid, cells and session pool as the qwen3 index run, served a second time to llama3:latest (4,320 calls over both architectures); every row's model field verified llama3:latest

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index | 720 | 720 | 720 | 0 | 720 | single-turn | 864 | 1590 | 19.4% |
| index_nok | 720 | 720 | 720 | 0 | 720 | single-turn | 860 | 1585 | 19.3% |
| plain | 720 | 720 | 720 | 0 | 720 | single-turn | 842 | 1567 | 19.1% |

Total analysed 2160, total excluded 0. Largest prompt any arm reached: 1590 tokens (19.4% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Rewritable-index cost (5 arms)

*exclusion rule:* turns 7--29 of the published cost sessions (window full), the identical pre-declared window as the primary three-arm cost run; byte-identical shared head asserted at generation

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index_I2 | 460 | 460 | 460 | 0 | 460 | 7--29 | 481 | 671 | 8.2% |
| index_I24 | 460 | 460 | 460 | 0 | 460 | 7--29 | 650 | 922 | 11.3% |
| index_I8 | 460 | 460 | 460 | 0 | 460 | 7--29 | 545 | 741 | 9.0% |
| length_only | 460 | 460 | 460 | 0 | 460 | 7--29 | 1591 | 1764 | 21.5% |
| prefix_stable | 460 | 460 | 460 | 0 | 460 | 7--29 | 453 | 641 | 7.8% |

Total analysed 2300, total excluded 0. Largest prompt any arm reached: 1764 tokens (21.5% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Token-compressor competitor (7 arms)

*exclusion rule:* turns 7--29 of the published cost sessions (window full), the identical pre-declared window as the primary three-arm cost run; every compressed prompt asserted at generation to be a whitespace-token subsequence of the same trial's naive window, with the probe tail byte-identical. Roster: naive-window control + 4 open keep-rates (.50/.25/.20/.12) + 2 gold-protected rates (.25/.20); the .20 pair is the budget-matched arm added mid-leg under AMENDMENT 2, served under the leg's original nonce so its prompts are the gated bytes

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comp_keep20 | 552 | 552 | 552 | 0 | 552 | 7--29 | 443 | 607 | 7.4% |
| comp_keep25 | 552 | 552 | 552 | 0 | 552 | 7--29 | 529 | 741 | 9.0% |
| comp_open12 | 552 | 552 | 552 | 0 | 552 | 7--29 | 292 | 371 | 4.5% |
| comp_open20 | 552 | 552 | 552 | 0 | 552 | 7--29 | 434 | 598 | 7.3% |
| comp_open25 | 552 | 552 | 552 | 0 | 552 | 7--29 | 526 | 732 | 8.9% |
| comp_open50 | 552 | 552 | 552 | 0 | 552 | 7--29 | 988 | 1196 | 14.6% |
| length_only | 552 | 552 | 552 | 0 | 552 | 7--29 | 1585 | 1860 | 22.7% |

Total analysed 3864, total excluded 0. Largest prompt any arm reached: 1860 tokens (22.7% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Token-compressor competitor, 2nd model

*exclusion rule:* same sessions, turns, arm roster and nonce discipline as the primary competitor leg, served to the second published cost architecture

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comp_keep20 | 552 | 552 | 552 | 0 | 552 | 7--29 | 453 | 617 | 7.5% |
| comp_keep25 | 552 | 552 | 552 | 0 | 552 | 7--29 | 539 | 751 | 9.2% |
| comp_open12 | 552 | 552 | 552 | 0 | 552 | 7--29 | 302 | 381 | 4.7% |
| comp_open20 | 552 | 552 | 552 | 0 | 552 | 7--29 | 444 | 608 | 7.4% |
| comp_open25 | 552 | 552 | 552 | 0 | 552 | 7--29 | 536 | 742 | 9.1% |
| comp_open50 | 552 | 552 | 552 | 0 | 552 | 7--29 | 998 | 1206 | 14.7% |
| length_only | 552 | 552 | 552 | 0 | 552 | 7--29 | 1595 | 1870 | 22.8% |

Total analysed 3864, total excluded 0. Largest prompt any arm reached: 1870 tokens (22.8% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Shared-key index fidelity, qwen3 (2 arms)

*exclusion rule:* every designed cell (H in {6,24} x first/middle/last) issued once per arm over the frozen gold-position session pool, with the distractor re-keyed to TARGET so a stale value competes under one key (the C7 shared-key hazard served back on top of the published STATE); the grid holds 2 arms (index fresh, index_nok stale) x 720 cells = 1,440 trials and the row's own arm column splits them; the row's model field is verified qwen3:14b, no post-hoc exclusion

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index | 720 | 720 | 720 | 0 | 720 | single-turn | 925 | 1675 | 20.4% |
| index_nok | 720 | 720 | 720 | 0 | 720 | single-turn | 927 | 1677 | 20.5% |

Total analysed 1440, total excluded 0. Largest prompt any arm reached: 1677 tokens (20.5% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Shared-key index fidelity, llama3 (2 arms)

*exclusion rule:* same grid, cells, shared-key hazard and session pool as the qwen3 shared-key index run, served a second time to llama3:latest (2,880 calls over both architectures); every row's model field verified llama3:latest

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index | 720 | 720 | 720 | 0 | 720 | single-turn | 862 | 1587 | 19.4% |
| index_nok | 720 | 720 | 720 | 0 | 720 | single-turn | 864 | 1589 | 19.4% |

Total analysed 1440, total excluded 0. Largest prompt any arm reached: 1589 tokens (19.4% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Repeat-variance diagnostic (2 arms)

*exclusion rule:* DIAGNOSTIC, not a confirmatory study: turns 7--29 of sessions S0/S5 of the primary cost run, re-fired three times per arm with a fresh nonce each repeat; generated cells 2 sessions x (23 turns x 3 repeats), all issued, none excluded

| arm | generated | designed cells | calls issued | server errors | analysed | turns used | mean prompt tok | max prompt tok | max/num_ctx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_only | 138 | 138 | 138 | 0 | 138 | 7--29 | 1608 | 1732 | 21.1% |
| prefix_stable | 138 | 138 | 138 | 0 | 138 | 7--29 | 433 | 614 | 7.5% |

Total analysed 276, total excluded 0. Largest prompt any arm reached: 1732 tokens (21.1% of the pinned budget), so no arm could have been dropped for overflowing the context window.

## Passage-danger audit (the multiplication behind the roster row)

| pool served per stratum | strata | calls | = | paired questions | primary slice (>=0.60) | sensitivity (>=0.55) |
| --- | --- | --- | --- | --- | --- | --- |
| 300 questions x 2 arms | 6 | 3600 | 6x2x300 | 1800 | 282 | 444 |
| 74 questions x 2 arms | 6 | 888 | 6x2x74 | 444 | 282 | 444 |
| 44 questions x 2 arms | 2 | 176 | 2x2x44 | 88 | 88 | 88 |

total 4664 calls in 14 files; 652 pairs (2 arms each) in the >=0.60 primary slice = 1304 calls carrying the C4 contrasts, and 3360 calls served outside it (the pool the slice is drawn from, reported as prevalence in the threshold table and entering no contrast).

## Global audit

34410 model calls across 39 run files; largest prompt anywhere 1870 tokens (22.8% of num_ctx) in prefix_comp_scored_qwen2.5_14b_t0.0.jsonl.

Of these, 34134 calls in 38 run files carry the confirmatory analyses reported in the paper, and 276 calls in 1 file (ms_repeat_variance_scored.jsonl) are a measurement-validity replication counted in the audit but supporting no inferential claim.

Legs of the confirmatory total that a reader is most likely to want rebuilt: gold-position stress 1440 calls in 2 files; rewritable-index fidelity 4320 calls in 2 files (the same 2,160-cell grid served to 2 architectures); rewritable-index cost 2300 calls in 1 files; token-compressor competitor 7728 calls in 2 files; shared-key index fidelity 2880 calls in 2 files (the same 720-cell position grid, distractor re-keyed TARGET so a stale value competes under one key, served to 2 architectures).
