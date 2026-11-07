# Enable-time anchor choices (CPU replay)

A is the default. Frozen boot observations only: A = 8192, B = 8192 + 32768. Each model row covers 30 cells; together the rows cover all 60. Predictions use actual ART scalar pricing and MAX over eight independently fitted rank-local profiles, without learning held-out cells. Raw excludes safety and cold; effective includes both. The reserve is already subtracted in the recorded usable limit.

| Option | Model | Added boot seconds, estimated | All cells effective/measured | 65536 | 131072 | Memory-refused fitting 100K+ |
|---|---|---:|---|---|---|---|
| A | dense | 416.2 | 1.100000–7.335704× | 1.823198–5.976866× | 1.572616–4.964644× | 0/6 |
| A | MoE | 827.1 | 1.100000–3.427388× | 1.758811–2.730737× | 1.683278–2.615303× | 0/6 |
| B | dense | 464.5 | 1.100000–5.976866× | 1.669388–5.976866× | 1.438856–4.964644× | 0/6 |
| B | MoE | 926.8 | 1.100000–1.903336× | 1.795886–1.903336× | 1.718761–1.772755× | 0/6 |
| C-proxy | dense | 0.0 | 1.100000–5.976866× | 1.669388–5.976866× | 1.438856–4.964644× | 0/6 |
| C-proxy | MoE | 0.0 | 1.100000–1.903336× | 1.795886–1.903336× | 1.718761–1.772755× | 0/6 |

A refuses none of the twelve measured-fitting 131K roots on the requested memory-only criterion (`effective > usable`), so B is not selected. The new held-out-anchor tests cover both options and permutations. Without cold extrapolation after anchoring, A minima would be 0.898638 dense / 0.961873 MoE, an unsafe outcome. Cold remains 1.75 outside the observed extent; anchored profiles price NEW, unanchored profiles price max(OLD envelope, NEW).

C interpretation: configured warm ranks using the original single-root ladder, without added RL anchors. Its available-rung projection uses only 8K/32K: the fixture does not measure 512, 2048 or 81920. The C-proxy numbers are not an exact enabled boot estimate, not a proof for MoE ordinary RL signatures (which differ from its single-target backward caller signature), and do not qualify the missing rungs. No numerical result is invented for those observations. Default configured ranks are 1,8,32; a narrower rank list also leaves the excluded ranks unanchored and is not covered by the C-proxy row.

Timing: estimates use run6 MoE 8K/32K forward/backward GPU durations, v7 dense durations, 0.9 s planning per new RL rung, and one first-8K-call allowance per output family (2 dense / 6 MoE). They assume graph reuse across ranks and conservatively do not subtract the legacy root rungs removed by A. A adds 12/36 RL rungs; B adds 24/72. Hidden/route replay and multi-root first-use timing remain unmeasured. If cold compilation repeats per rank, A additions are 1192.2/2308.5 s dense/MoE. Baseline is approximately 780 s against CALIBRATION_SECONDS=3600; these estimates do not guarantee a deadline.

Important scope: the replay has metadata-only scalar floors, not all native operator-specific floor inputs, which the fixture does not serialize. Root peaks stand in for anchor observations; new 64-root peaks must be measured on GPU and may differ. A retains an actual calibrated context bound of 8K in Caladan; v9 calls ART directly past that bound just as v7 did. The existing 8x profile trust window is unchanged. Numerical 131K and 262K prices from frozen 8K evidence do not assert trust; real progressive waves may extend observations within that window. The 262K rows are extrapolation only, with no measured peak or usable limit and no admission guarantee.

Reproduce: `dev/memory_fixed_slope/cpu.sh python dev/memory_fixed_slope/anchor_choices.py`. Full per-cell raw/effective bytes, ratios, usable limits, cold factor and trust flags are in `anchor-choices.csv` and `anchor-choices.json`.
