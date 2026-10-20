"""Lossless conversion and public Tinker exporter compatibility fixtures."""

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import torch

from art.megatron.weights.compact_lora import (
    FORMAT,
    compact_to_vllm,
    load_compact,
    pack_compact_gradients,
    pack_serving_tensors,
    per_expert_to_shared,
    refactor_per_expert_to_shared,
    save_compact,
    shared_to_per_expert,
    tinker_expert_tensors,
)


class CompactTests(unittest.TestCase):
    def test_exact_replica_roundtrip(self):
        for factor in ("A", "B"):
            a = torch.randn(1 if factor == "A" else 3, 16, 8)
            b = torch.randn(1 if factor == "B" else 3, 8, 12)
            expanded = shared_to_per_expert(a, b, experts=3)
            compact = per_expert_to_shared(*expanded, shared_factor=factor)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip((a, b), compact)))
            changed = [t.clone() for t in expanded]
            changed[0 if factor == "A" else 1][1, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "losslessly"):
                per_expert_to_shared(*changed, shared_factor=factor)

    def test_gauge_conversion_and_rank_rejection(self):
        for factor in ("A", "B"):
            a, b = (
                torch.randn(1 if factor == "A" else 3, 16, 2, dtype=torch.float64),
                torch.randn(1 if factor == "B" else 3, 2, 12, dtype=torch.float64),
            )
            ae, be = shared_to_per_expert(a, b, experts=3)
            # Per-expert gauge transformations preserve the original update.
            for e in range(3):
                ae[e] *= e + 1
                be[e] /= e + 1
            ca, cb = refactor_per_expert_to_shared(ae, be, shared_factor=factor, rank=2)
            torch.testing.assert_close(ca @ cb, ae @ be)
            with self.assertRaisesRegex(ValueError, "rank or tolerance"):
                refactor_per_expert_to_shared(ae, be, shared_factor=factor, rank=1)
            ca, cb = refactor_per_expert_to_shared(
                ae, be, shared_factor=factor, rank=4, source_scale=16, target_scale=8
            )
            torch.testing.assert_close((ca @ cb) * 8, (ae @ be) * 16)

    def test_exact_converter_preserves_replica_bits(self):
        a, b = torch.zeros(3, 16, 8), torch.randn(3, 8, 12)
        a[1, 0, 0] = -0.0
        with self.assertRaisesRegex(ValueError, "losslessly"):
            per_expert_to_shared(a, b, shared_factor="A")

    def test_canonical_arena_reuses_storage_and_matches_sorted_wire(self):
        factors = {
            "z": torch.randn(3, 8, 12).bfloat16(),
            "a": torch.randn(1, 16, 8).bfloat16(),
        }
        out = torch.empty(
            sum(t.numel() * 2 for t in factors.values()), dtype=torch.uint8
        )
        arena, metadata = pack_serving_tensors(factors, out=out)
        self.assertEqual(arena.data_ptr(), out.data_ptr())
        self.assertEqual([e["name"] for e in metadata["tensors"]], ["a", "z"])
        for entry in metadata["tensors"]:
            view = (
                arena[entry["offset"] : entry["offset"] + entry["nbytes"]]
                .view(torch.bfloat16)
                .reshape(entry["shape"])
            )
            self.assertTrue(torch.equal(view, factors[entry["name"]]))
        with self.assertRaisesRegex(ValueError, "BF16"):
            pack_serving_tensors({"a": factors["a"].float()})

    def test_fp32_root_arena_keeps_shared_extent_and_reuses_storage(self):
        from types import SimpleNamespace

        a = torch.nn.Parameter(torch.randn(1, 16, 8, dtype=torch.bfloat16))
        b = torch.nn.Parameter(torch.randn(3, 8, 12, dtype=torch.bfloat16))
        a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
        module = SimpleNamespace(
            adapter_model_prefix="experts.{expert}.gate_up_proj",
            _lora_params=lambda ref: [("A", a), ("B", b)],
        )
        out = torch.empty(a.numel() + b.numel(), dtype=torch.float32)
        arena, metadata = pack_compact_gradients([module], out=out)
        self.assertEqual(arena.data_ptr(), out.data_ptr())
        self.assertEqual(
            arena.numel() * arena.element_size(), 4 * (a.numel() + b.numel())
        )
        torch.testing.assert_close(
            arena, torch.cat((a.grad.float().flatten(), b.grad.float().flatten()))
        )
        self.assertEqual(metadata["entries"][0]["shape"], [1, 16, 8])
        with self.assertRaisesRegex(ValueError, "geometry"):
            pack_compact_gradients([module], out=torch.empty(out.numel() + 1))

    def test_manifest_roundtrip(self):
        tensors = {"a": torch.randn(1, 16, 8), "b": torch.randn(3, 8, 12)}
        manifest = {"format": FORMAT, "modules": [{"keys": ["a", "b"]}]}
        with tempfile.TemporaryDirectory() as path:
            save_compact(path, tensors, manifest)
            loaded, got = load_compact(path)
            self.assertEqual(got, manifest)
            self.assertTrue(all(torch.equal(tensors[k], loaded[k]) for k in tensors))
            bad = Path(path) / "manifest.json"
            bad.write_text(json.dumps({**manifest, "format": "unknown"}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_compact(path)

    def test_compact_bridge_refuses_independent_and_sharded_inputs(self):
        from types import SimpleNamespace

        handler = SimpleNamespace(
            to_vllm_lora_tensors=lambda tensors, adapter_config: (
                dict(tensors),
                dict(adapter_config),
            )
        )
        tensors = {
            "ga": torch.randn(1, 16, 8),
            "gb": torch.randn(3, 8, 24),
            "da": torch.randn(3, 12, 8),
            "db": torch.randn(1, 8, 16),
        }
        prefix = "base_model.model.model.layers.0.mlp.experts.{expert}"
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "modules": [
                {
                    "prefix": prefix + ".gate_up_proj",
                    "keys": ["ga", "gb"],
                    "experts": [0, 1, 2],
                    "shared_factor": "A",
                },
                {
                    "prefix": prefix + ".down_proj",
                    "keys": ["da", "db"],
                    "experts": [0, 1, 2],
                    "shared_factor": "B",
                },
            ],
        }
        got, config = compact_to_vllm(
            tensors, manifest, handler=handler, adapter_config={"target_modules": []}
        )
        self.assertEqual(len(got), 6)
        self.assertEqual(config["stark_moe_layout"], "shared_outer_v1")
        self.assertEqual(
            sum(t.numel() for t in got.values()),
            sum(t.numel() for t in tensors.values()) + tensors["ga"].numel(),
        )
        manifest["modules"][0]["shared_factor"] = None
        with self.assertRaisesRegex(ValueError, "independent"):
            compact_to_vllm(
                tensors,
                manifest,
                handler=handler,
                adapter_config={"target_modules": []},
            )
        manifest["modules"][0]["shared_factor"] = "A"
        manifest["modules"][0]["experts"] = [3, 4, 5]
        with self.assertRaisesRegex(ValueError, "EP1"):
            compact_to_vllm(
                tensors,
                manifest,
                handler=handler,
                adapter_config={"target_modules": []},
            )

    def test_dense_bridge_makes_transposed_factors_contiguous(self):
        from types import SimpleNamespace

        handler = SimpleNamespace(
            to_vllm_lora_tensors=lambda tensors, adapter_config: (
                dict(tensors),
                dict(adapter_config),
            )
        )
        tensors = {"a": torch.randn(16, 8), "b": torch.randn(8, 12)}
        manifest = {
            "format": FORMAT,
            "modules": [
                {
                    "prefix": "q_proj",
                    "keys": ["a", "b"],
                    "experts": None,
                    "shared_factor": None,
                }
            ],
        }
        got, _ = compact_to_vllm(
            tensors,
            manifest,
            handler=handler,
            adapter_config={"target_modules": ["q_proj"]},
        )
        self.assertTrue(all(t.is_contiguous() for t in got.values()))
        torch.testing.assert_close(got["q_proj.lora_A.weight"], tensors["a"].T)

    def test_public_expert_namespace(self):
        a, b = torch.randn(1, 16, 8), torch.randn(3, 8, 24)
        da, db = torch.randn(3, 12, 8), torch.randn(1, 8, 16)
        tensors = tinker_expert_tensors(
            "base_model.model.model.layers.0.mlp.experts", a, b, da, db
        )
        self.assertEqual(len(tensors), 6)
        for e in range(3):
            for w, at, bt in (
                ("w1", a[0], b[e, :, :12]),
                ("w3", a[0], b[e, :, 12:]),
                ("w2", da[e], db[0]),
            ):
                p = f"base_model.model.model.layers.0.mlp.experts.{w}"
                actual = (
                    tensors[p + ".lora_B.weight"][0 if w == "w2" else e]
                    @ tensors[p + ".lora_A.weight"][e if w == "w2" else 0]
                )
                torch.testing.assert_close(actual, (at @ bt).T)


if __name__ == "__main__":
    unittest.main()
