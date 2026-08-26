from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import art
from art.distributed import (
    ArtLaunchContext,
    ArtRuntime,
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceSpec,
    NcclTransportSpec,
    NixlTransportSpec,
    TrainerMeshSpec,
    VllmParallelSpec,
    compile_topology,
)
from art.megatron.backend import MegatronBackend


async def main(launch: ArtLaunchContext) -> None:
    model_name = "multinode-release-smoke"
    base_model = os.environ.get("ART_EXAMPLE_MODEL", "Qwen/Qwen3-0.6B-Base")
    artifact_root = Path(os.environ["ART_SHARED_ROOT"]).expanduser().resolve()
    ranks_per_host = int(os.environ.get("ART_TRAINER_RANKS_PER_HOST", "1"))
    nccl_net = os.environ.setdefault("NCCL_NET", "IB")
    cluster = launch.homogeneous_cluster(
        cpu_slots=2,
        gpu_ids=tuple(range(ranks_per_host)),
        artifact_root=str(artifact_root),
        cache_root="/tmp/art-cache",
        nccl_transport=NcclTransportSpec(net_name=nccl_net),
        nixl_transport=(
            NixlTransportSpec()
            if os.environ.get("ART_EXAMPLE_USE_NIXL") == "1"
            else None
        ),
        startup_timeout_s=900,
    )
    topology = art.MegatronTopologyConfig(
        tp=int(os.environ.get("ART_EXAMPLE_TP", "1")),
        cp=int(os.environ.get("ART_EXAMPLE_CP", "1")),
        ep=int(os.environ.get("ART_EXAMPLE_EP", "1")),
        pp=int(os.environ.get("ART_EXAMPLE_PP", "1")),
    )
    art.init_megatron_runtime_config(
        topology=topology,
        packed_sequence_length=512,
    )
    leader_host = urlparse(cluster.hosts[0].worker_address).hostname
    if leader_host is None:
        raise RuntimeError("controller worker address has no host")
    model_service = ModelServiceSpec(
        name=model_name,
        members=(
            ModelServiceMemberSpec(
                member_id="inference",
                host_id=cluster.hosts[0].host_id,
                node_rank=0,
                gpu_ids=(cluster.hosts[0].gpu_ids[0],),
            ),
        ),
        leader_endpoint=EndpointSpec(host=leader_host, port=8000),
        rendezvous=EndpointSpec(host=leader_host, port=29500),
        base_model=base_model,
        runtime_fingerprint=hashlib.sha256(base_model.encode()).hexdigest(),
        parallel=VllmParallelSpec(),
        temporal_gpu_sharing=True,
    )
    runtime = await ArtRuntime.start(
        launch.host_mesh,
        compile_topology(
            cluster=cluster,
            rollout_host_ids=(),
            trainer=TrainerMeshSpec(
                ranks=cluster.gpu_placements(),
                topology=topology,
            ),
            model_services=(model_service,),
        ),
    )
    try:
        async with MegatronBackend(runtime=runtime) as backend:
            model = art.TrainableModel(
                name=model_name,
                project="art-release",
                run_name="multinode-release-smoke",
                base_model=base_model,
                _internal_config={"init_args": {"max_seq_length": 512}},
                report_metrics=[],
            )
            await model.register(backend)
            await model.train_sft(
                [
                    art.Trajectory(
                        messages_and_choices=[
                            {"role": "user", "content": "Answer yes."},
                            {"role": "assistant", "content": "Yes."},
                        ],
                        reward=1.0,
                    ),
                    art.Trajectory(
                        messages_and_choices=[
                            {"role": "user", "content": "Answer no."},
                            {"role": "assistant", "content": "No."},
                        ],
                        reward=1.0,
                    ),
                ],
                art.TrainSFTConfig(learning_rate=1e-6, batch_size=2),
                log_metrics=False,
            )
            print(
                f"ART_MULTINODE_TRAIN_PASS hosts={launch.host_count} "
                f"ranks={len(cluster.gpu_placements())} step={await model.get_step()}",
                flush=True,
            )
    finally:
        await runtime.close()
