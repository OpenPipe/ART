from __future__ import annotations

import json
import unittest

from art_vllm_runtime import binary_routes
import numpy as np

from art.vllm_route_transport import decode_routed_experts_response


class BinaryRoutesProtocolTest(unittest.TestCase):
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
                "num_experts_per_tok": 2,
                "mlp_layer_types": ["dense", "sparse"],
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

    def test_does_not_compute_route_conflicts(self) -> None:
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=(0, 1, 2))
        values = np.zeros((1, 5, 2), dtype=np.uint8)
        values[:, 3, :] = (2, 5)
        routes[0] = values

        binary_routes.encode_routed_experts_response(b"{}", routes)

    def test_rejects_route_outside_model_range(self) -> None:
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=())
        routes[0] = np.asarray([[[0, 8]]], dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "outside the exact model range"):
            binary_routes.encode_routed_experts_response(b"{}", routes)


if __name__ == "__main__":
    unittest.main()
