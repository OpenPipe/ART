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

    def test_capture_registers_vllm_authoritative_expert_count(self) -> None:
        model_config = type("ModelConfig", (), {"get_num_experts": lambda _self: 257})()
        previous = binary_routes._REGISTERED_NUM_EXPERTS
        try:
            binary_routes._REGISTERED_NUM_EXPERTS = None
            binary_routes._register_model_num_experts(model_config)
            with binary_routes.capture_routed_experts() as routes:
                self.assertEqual(routes.num_experts, 257)
        finally:
            binary_routes._REGISTERED_NUM_EXPERTS = previous


if __name__ == "__main__":
    unittest.main()
