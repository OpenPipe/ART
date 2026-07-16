# Multi-Replica Preservation Gate

This directory preserves the optional independent-deployment routing layer before
the core multi-node branch is simplified to one logical model deployment. Native
vLLM TP/PP/DP within that deployment is not part of this split.

## Focused Gate

```bash
.venv/bin/pytest -q \
  scratch/test_vllm_gateway.py \
  scratch/test_vllm_gateway_resilience.py \
  scratch/test_vllm_kv_events.py \
  scratch/test_vllm_prefix_router.py \
  scratch/test_vllm_replica_ack.py \
  scratch/test_replica_recovery.py \
  scratch/test_distributed_inference_metrics.py
```

`live_two_replica_prefix_affinity.py` validates two independently routed TP4
deployments. `live_validation_20260715/prefix_affinity_fixed2.json` records both
repeated 8,192-token requests selecting replica 1, 8,128 cached tokens on the
second request, and zero prompt tokens on replica 0.

`live_vllm_member_recovery.py` validates whole-deployment failure fencing and
restart. `vllm_recovery_result.json` records generation 0 to 1 replacement and
successful inference before and after restart.

`benchmark_vllm_router.py` is the bounded CPU routing-overhead probe. These files
are branch evidence, not production package APIs.
