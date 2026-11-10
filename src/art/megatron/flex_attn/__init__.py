"""ART Megatron flex-attention integration."""

from .flash_pdl_patch import apply_flash_sm90_sparse_pdl_patch

# Establish the dependency boundary before compiled.py can lower or capture it.
apply_flash_sm90_sparse_pdl_patch()
