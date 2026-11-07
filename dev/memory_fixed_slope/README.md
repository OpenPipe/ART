# CPU memory-model prototype artifacts

See ../../DESIGN.md for the approved Phase 1 design and PHASE-2.md for the current cold guard, migration, coverage and estimates. No GPU execution is part
of these scripts. All GPU measurements are supplied fixtures. `COMPARISON.md`
and `comparison.csv` contain every OLD/NEW raw and effective ratio; `RUN6.md`
separates refusals from actual fit evidence. `wave.json` records the approximate
568-datum dense RL what-if (24 GiB usable, CP8, LoRA 8).

The worktree-local `.venv` has the exact installed CPU packages recorded in
`cpu-requirements.txt`. To rebuild it:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  -r dev/memory_fixed_slope/cpu-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
```
There is no editable install of another worktree.

Run from the ART worktree:

```sh
python3 dev/memory_fixed_slope/analyze_old.py
dev/memory_fixed_slope/cpu.sh python dev/memory_fixed_slope/replay.py
dev/memory_fixed_slope/cpu.sh python dev/memory_fixed_slope/wave.py
dev/memory_fixed_slope/cpu.sh python -m pytest tests/unit/test_trainer_rank_fixed_slope.py -q
```

`cpu_imports/sitecustomize.py` bypasses unused public service initializers and
lifts unchanged provider helper definitions from ART to avoid CUDA-heavy training
imports. `cpu_lora.py` lifts the real LoRA metadata class definitions, using meta
factors without TE/model construction. `cpu_planner.py` follows the host-overhead
lane's harness. Planner search, packing, memory pricing and width selection are
ART implementations. These shims are for CPU experiments, not native execution
qualification.

The Caladan companion [#2706](https://github.com/OpenPipe/caladan/pull/2706) on
`stark/memory-fixed-slope-v2` contains
the real v4 key/merge, default-off flag and multi-segment calibration code. It
supersedes the Phase 1 sample patch and temporary-codec checker, which remain
available in commit 36a46dc27's history. See PHASE-2.md for coupling requirements.
The enabled default is A (8K only), with cold extrapolation after anchoring;
`ANCHOR-CHOICES.md` and `anchor_choices.py` give the frozen-anchor choice replay.

The original poison regression failed before implementation; its transcript
is `poisoning-before.txt`. `agreement-parity.json` records unchanged agreement,
trust and refusal-path source. Review `VALIDATION.md` for test and check limits.
