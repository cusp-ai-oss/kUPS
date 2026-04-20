# Troubleshooting

## GPU Utilization

GPU utilization may be low when computing a single small system. JAX launches kernels asynchronously, but if each kernel is small relative to the GPU's capacity, the hardware sits idle between launches. Running multiple independent systems in parallel (e.g., via batching with `Table.union` or `jax.vmap`) keeps more cores busy and improves throughput.

Monitor utilization with `nvidia-smi` to check whether your workload is compute-bound or launch-bound.

## GPU Memory

JAX preallocates GPU memory at startup to avoid fragmentation and reduce allocation overhead. When you check `nvidia-smi`, it may show high memory usage even when your computation is not using all of it. This is expected.

The memory shown represents JAX's preallocated pool, not the amount actively in use. This strategy improves performance by avoiding frequent allocations and deallocations during the simulation.

To share GPU memory with other processes or reduce the preallocated amount, configure JAX's memory allocation as described in the [JAX GPU memory allocation documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html). Be aware that disabling preallocation may reduce performance.

## Common Errors

| Error | Cause and Solution |
| -- | -- |
| `XlaRuntimeError: UNAVAILABLE: Parallel compilation was requested, but no available compilation provider supports it` or `XlaRuntimeError: INTERNAL: RET_CHECK failure` | Most likely caused by a stale compilation cache. Delete the cache directory and retry. Also ensure that all necessary CUDA libraries are installed on the host in addition to JAX's PyPI CUDA packages. |
| `cusparseGetProperty ... The cuSPARSE library was not found.` | JAX picks up a version mismatch between the virtual environment's CUDA libraries and system-installed ones. Run `unset LD_LIBRARY_PATH` before starting the simulation to let JAX use its own bundled libraries. |
| `jax.errors.ConcretizationTypeError` | A value that depends on runtime data was used where JAX needs a compile-time constant. Common causes: using a traced value as an array shape, indexing with a traced value into a Python list, or branching on a traced boolean. See the [JAX sharp bits documentation](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html). |
| `jax.errors.TracerBoolConversionError` | A JAX tracer was used in a Python `if` statement or `bool()` call. JAX cannot trace through Python control flow that depends on array values. Use `jax.lax.cond` instead, or restructure the logic to avoid data-dependent branching. |
