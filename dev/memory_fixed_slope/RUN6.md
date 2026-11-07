# Run6 refusals and fit evidence

All byte columns are admission increments. Effective is the recorded safety-scaled operand; raw = effective / 1.10. Rank-zero records are preserved in the fixture. MiB/GiB below use powers of two.

| Name | Tokens | LoRA | Raw GiB | Effective GiB | Usable GiB | Real-fit evidence |
|---|---:|---:|---:|---:|---:|---|
| calibration/forward_backward | 81920 | 8 | 64.739105 | 71.213016 | 61.103647 | No same-run measured peak; root-only sweep does not validate this shape. |
| calibration/forward_backward | 32768 | 32 | 74.902924 | 82.393216 | 54.027475 | No same-run measured peak; root-only sweep does not validate this shape. |
| wanda | 2944 | 8 | 106.406898 | 117.047588 | 61.086069 | No same-run measured peak; root-only sweep does not validate this shape. |
| forward-8192 | 8192 | 8 | 481.207031 | 529.327734 | 61.086069 | Same-run calibration/forward at 81920 succeeded at LoRA 8 before refusal; no peak bytes logged. |
| forward-32768 | 32768 | 8 | 1924.828125 | 2117.310937 | 61.086069 | Same-run calibration/forward at 81920 succeeded at LoRA 8 before refusal; no peak bytes logged. |
| numerics/forward_backward | 2944 | 8 | 106.406898 | 117.047588 | 60.136850 | No same-run measured peak; root-only sweep does not validate this shape. |

The 8192 and 32768 refusals are exactly proportional (up to integer truncation). Their effective rate is about 66.17 MiB/token (69.38 decimal MB/token); this differs from the brief's 64.6 MB/token shorthand. The test and reports use the exact stored byte operands. The brief's “529 GB / 2117 GB” labels are GiB in the error message. Successful 81920 forwards are existence evidence, not a numerical peak for either refused shape. Wanda was not executed after refusal; do not invent its measured peak or a predicted/measured ratio.

The v7 8192 and 32768 MoE forward observations (different run/model state) measured 771460608 and 2058245632 bytes. They support investigating pessimistic admission but are not a substitute for same-run Wanda/target-shape qualification.

Wanda's refusal operand prices the minimum 2944-token datum after admission search exhausted splitting; the original request had 16 such datums (47104 logical tokens, expected shared packed count 12544). It is not a measured peak of that full wave.
