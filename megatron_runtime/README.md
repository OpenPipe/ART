# ART Megatron runtime

This private lock project defines the trainer environment materialized by ART on
each trainer host. It is bundled into release wheels and is not a separately
published package.

The root package owns orchestration and lightweight CPU services. Monarch starts
trainer ranks with this runtime's Python executable, keeping Megatron's compiled
dependency stack exact without exposing source-only dependencies through ART's
published wheel metadata.

The CUDA-specific profiles own different layers of the runtime:

- PyTorch, Megatron Core/Bridge, Transformer Engine, FlashAttention, and the
  model kernels are locked here.
- CUDA 12 builds pinned NVIDIA Apex with its CUDA extensions because Megatron's
  gradient-accumulation fusion imports `fused_weight_gradient_mlp_cuda`. CUDA 13
  uses ART's explicitly unfused provider path and does not install Apex.
- The official `nixl-cu12` and `nixl-cu13` wheels provide Python bindings,
  `libnixl`, and relocatable UCX libraries/plugins. ART's release wheel embeds
  the matching NIXL and UCX source headers required to compile HybridEP.
- ART builds its own HybridEP source only for trainer topologies that use EP.
  Cross-host EP additionally validates the image's GDA/RDMA capabilities before
  compilation.

The supported cluster image still owns the NVIDIA driver and CUDA toolkit,
native build tools, NCCL network transport, MOFED/RDMA devices, and required
kernel modules. A Python package cannot install or safely replace those host
capabilities.
