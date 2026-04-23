"""Fast reimplementations of the MMA-Sim oracle.

Each submodule exposes `mma` (and eventually `mma_block_scale`) classes
with the same signature as `mmasim.simulator.nv_ptx.mma`, but implemented
without per-element Python loops.

- numpy_ref: vectorized NumPy reference (Phase 1).
"""
