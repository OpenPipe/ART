"""CPU signature-key inventory; model metadata stand-ins, no GPU qualification."""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from cpu_planner import make_rank
import torch

from art.trainer_rank import ForwardInput, _memory


def requests(count, family):
    out = []
    for i in range(count):
        tokens = torch.arange(128) + i * 128
        routes = (
            torch.zeros((128, 40, 8), dtype=torch.long)
            if "routes" in family and ("mixed" not in family or i % 2)
            else None
        )
        out.append(
            ForwardInput(
                input_tokens=tokens,
                target_tokens=None if family.startswith("hidden") else tokens,
                hidden_states=family.startswith("hidden"),
                routed_experts=routes,
            )
        )
    return out


def main():
    records = []
    for model in ("dense", "moe"):
        r = make_rank()
        if model == "moe":
            r._moe_layers = 40
            r._moe_memory_supported = True
        # Assert the supported native checkpoint gate for this signature-only
        # experiment. The stand-in model is not a native MoE operator graph.
        with patch.object(
            _memory, "_checkpoint_layers", return_value=40 if model == "moe" else 64
        ):
            for grad in (False, True):

                def sig(req):
                    return r._memory_signature_from_requests(
                        req,
                        slot_group_count=1,
                        grad_modes=(grad,),
                        slot_groups=((r._default_slot_ref, grad),),
                    )

                root = sig(requests(1, "target"))
                for family in ["target", "hidden"] + (
                    [
                        "target+routes",
                        "target-mixed-routes",
                        "hidden+routes",
                        "hidden-mixed-routes",
                    ]
                    if model == "moe"
                    else []
                ):
                    wave = sig(requests(64, family))
                    a = asdict(root)
                    b = asdict(wave)
                    delta = {
                        k: {"root": a.get(k), "wave": b.get(k)}
                        for k in sorted(a.keys() | b.keys())
                        if a.get(k) != b.get(k)
                    }
                    records.append(
                        dict(
                            model=model,
                            operation="forward_backward" if grad else "forward",
                            family=family,
                            root_type=type(root).__name__,
                            wave_type=type(wave).__name__,
                            different_fields=delta,
                            same_with_256_segments=wave == sig(requests(256, family)),
                        )
                    )
    p = Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(records, indent=2) + "\n")
    for r in records:
        print(
            r["model"],
            r["operation"],
            r["family"],
            r["root_type"],
            r["wave_type"],
            list(r["different_fields"]),
        )


if __name__ == "__main__":
    main()
