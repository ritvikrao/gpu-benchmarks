# gpu-benchmarks
Benchmarks for gpu codes written in Charm++ and Kokkos/Cuda/HIP

## Benchmarks

Each folder in `benchmarks/` is a separate benchmark. The initial benchmark is:

- `benchmarks/hapi-kernel-launch`: one-process Charm++ SMP benchmark with a single chare array where each chare launches Kokkos kernels on its own stream and chains launches via Charm++ HAPI callbacks.

## Building

Use CMake and pass paths to Charm++ and Kokkos:

```bash
cmake -S . -B build \
  -DCHARM_ROOT=/path/to/charm \
  -DKOKKOS_ROOT=/path/to/kokkos \
  -DBENCHMARK_BACKEND=CUDA
cmake --build build -j
```

Set `-DBENCHMARK_BACKEND=HIP` to build the HIP variant.

## Running

Example run (10 seconds by default):

```bash
./build/benchmarks/hapi-kernel-launch/hapi_kernel_launch +p4 +ppn 4 --chares-per-thread=1 --duration-seconds=10
```
