# CPU memory-model prototype artifacts

See ../../DESIGN.md for decisions, scope and results. No GPU execution is part
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
dev/memory_fixed_slope/cpu.sh python dev/memory_fixed_slope/check_caladan.py
dev/memory_fixed_slope/cpu.sh python -m pytest tests/unit/test_trainer_rank_fixed_slope.py -q
```

`cpu_imports/sitecustomize.py` bypasses unused public service initializers and
lifts unchanged provider helper definitions from ART to avoid CUDA-heavy training
imports. `cpu_lora.py` lifts the real LoRA metadata class definitions, using meta
factors without TE/model construction. `cpu_planner.py` follows the host-overhead
lane's harness. Planner search, packing, memory pricing and width selection are
ART implementations. These shims are for CPU experiments, not native execution
qualification.

`caladan-v4.patch` is a companion format/key/merge change for the Caladan base
recorded in `caladan-provenance.json`; it is not applied to that worktree.
`check_caladan.py` applies it to a temporary copy and exercises real codecs.
Coordinated startup/calibration collection activation is still required after
GPU review: the internal fixed-slope context is opt-in in this Phase 1 branch.

The original poison regression failed before implementation; its transcript
is `poisoning-before.txt`. `agreement-parity.json` records unchanged agreement,
trust and refusal-path source. Review `VALIDATION.md` for test and check limits.
