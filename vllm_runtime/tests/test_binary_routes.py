from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from art_vllm_runtime import binary_routes
import numpy as np

from art.vllm_route_transport import (
    decode_retained_route_bundle,
    decode_routed_experts_response,
    retained_route_bundle_from_response,
    route_bundle_id,
)


class BinaryRoutesProtocolTest(unittest.TestCase):
    def test_route_request_binds_private_marker_and_cache_identity(self) -> None:
        request = SimpleNamespace(vllm_xargs={"other": 2}, cache_salt="policy")

        binary_routes.mark_route_request(request)
        binary_routes.mark_route_request(request)

        self.assertEqual(
            request.vllm_xargs,
            {"other": 2, binary_routes.ROUTE_REQUEST_XARG: 1},
        )
        self.assertEqual(request.cache_salt, "policy|art-routes=ARTRTE2")

    def test_only_marked_scheduled_requests_enable_capture(self) -> None:
        marked = SimpleNamespace(extra_args={binary_routes.ROUTE_REQUEST_XARG: 1})
        ordinary = SimpleNamespace(extra_args=None)
        runner = SimpleNamespace(
            requests={"cached": SimpleNamespace(sampling_params=marked)}
        )
        output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="new", sampling_params=ordinary)
            ],
            num_scheduled_tokens={"new": 2},
        )
        self.assertFalse(binary_routes._scheduled_request_routes(runner, output))

        output.num_scheduled_tokens["cached"] = 1
        self.assertTrue(binary_routes._scheduled_request_routes(runner, output))

    def test_exact_expert_count_and_dtype_roundtrip(self) -> None:
        response = json.dumps(
            {
                "id": "route-test",
                "choices": [],
                "created": 0,
                "model": "test-model",
                "object": "chat.completion",
            }
        ).encode()
        for num_experts, dtype, values in (
            (256, np.uint8, [[[0, 255]]]),
            (257, np.uint16, [[[0, 256]]]),
        ):
            body = binary_routes.encode_routed_experts_response(
                response,
                {0: np.asarray(values, dtype=dtype)},
                num_experts=num_experts,
            )
            decoded_response, routes = decode_routed_experts_response(body)

            self.assertEqual(decoded_response.id, "route-test")
            self.assertEqual(routes[0].num_experts, num_experts)
            self.assertEqual(routes[0].dtype, np.dtype(dtype))
            np.testing.assert_array_equal(routes[0], values)

    def test_retained_bundle_preserves_exact_bytes_and_token_identity(self) -> None:
        response = json.dumps(
            {
                "id": "route-retained",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {"content": "ok", "role": "assistant"},
                        "output_token_ids": [13],
                    }
                ],
                "created": 0,
                "model": "test-model",
                "object": "chat.completion",
                "prompt_token_ids": [11, 12],
            }
        ).encode()
        values = np.asarray([[[1]], [[2]], [[3]]], dtype=np.uint8)
        body = binary_routes.encode_routed_experts_response(
            response, {0: values}, num_experts=8
        )

        decoded_response, bundle = retained_route_bundle_from_response(
            body,
            request_id="request",
            owner_id="tenant/run",
            model_identity="test-model@generation-3",
        )
        payload = b"".join(bundle.chunks)
        routes = decode_retained_route_bundle(
            bundle.layout, payload, token_ids={0: (11, 12, 13)}
        )

        self.assertEqual(decoded_response.id, "route-retained")
        self.assertEqual(bundle.layout.byte_count, values.nbytes)
        self.assertEqual(bundle.layout.bundle_id, route_bundle_id(bundle.layout))
        np.testing.assert_array_equal(routes[0], values)
        with self.assertRaisesRegex(RuntimeError, "token identity"):
            decode_retained_route_bundle(
                bundle.layout, payload, token_ids={0: (11, 12, 14)}
            )

    def test_rejects_expert_count_beyond_uint16_protocol(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"\[1, 65536\]"):
            binary_routes.encode_routed_experts_response(
                b"{}",
                {0: np.zeros((1, 1, 1), dtype=np.uint16)},
                num_experts=65_537,
            )

    def test_capture_registers_vllm_authoritative_route_layout(self) -> None:
        text_config = type(
            "TextConfig",
            (),
            {
                "num_hidden_layers": 2,
                "mlp_layer_types": ["dense", "sparse"],
                "num_experts_per_tok": 2,
            },
        )()
        model_config = type(
            "ModelConfig",
            (),
            {
                "get_num_experts": lambda _self: 257,
                "hf_text_config": text_config,
            },
        )()
        previous = (
            binary_routes._REGISTERED_NUM_EXPERTS,
            binary_routes._REGISTERED_PADDING_LAYERS,
        )
        try:
            binary_routes._REGISTERED_NUM_EXPERTS = None
            binary_routes._REGISTERED_PADDING_LAYERS = None
            binary_routes._register_model_route_layout(model_config)
            with binary_routes.capture_routed_experts() as routes:
                self.assertEqual(routes.num_experts, 257)
                self.assertEqual(routes.padding_layers, (0,))
        finally:
            (
                binary_routes._REGISTERED_NUM_EXPERTS,
                binary_routes._REGISTERED_PADDING_LAYERS,
            ) = previous

    def test_resolves_only_registered_padding_layers(self) -> None:
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=(0, 1, 2))
        values = np.zeros((2, 5, 2), dtype=np.uint8)
        values[:, 3, :] = (2, 5)
        values[:, 4, :] = (1, 7)
        routes[0] = values

        response = json.dumps(
            {
                "id": "route-test",
                "choices": [],
                "created": 0,
                "model": "test-model",
                "object": "chat.completion",
            }
        ).encode()
        body = binary_routes.encode_routed_experts_response(response, routes)
        _, decoded = decode_routed_experts_response(body)

        expected = np.broadcast_to((0, 1), (2, 3, 2))
        np.testing.assert_array_equal(decoded[0][:, :3, :], expected)
        np.testing.assert_array_equal(decoded[0][:, 3:, :], values[:, 3:, :])

    def test_rejects_missing_capture_on_routed_layer(self) -> None:
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=(0, 1, 2))
        values = np.zeros((1, 5, 2), dtype=np.uint8)
        values[:, 3, :] = (2, 5)
        routes[0] = values

        with self.assertRaisesRegex(RuntimeError, "must be distinct"):
            binary_routes.encode_routed_experts_response(b"{}", routes)


if __name__ == "__main__":
    unittest.main()
