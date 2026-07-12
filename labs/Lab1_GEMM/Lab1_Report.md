---
title: "ECDNN Lab 1: General Matrix Multiplication"
author: "Name: [Your Name]    Student ID: [Your Student ID]"
date: "July 2026"
---

## Experimental environment

- Hardware: Apple A18 Pro (arm64)
- Compiler: Apple Clang 21.0.0
- Build flags: `-std=c++17 -O3 -Wall`
- Timings: arithmetic time only, averaged over 32 runs. Input initialization and correctness checks are excluded from the timed region.

All matrix programs use `uint32_t` values. Unsigned overflow has defined wraparound semantics, so every loop order produces a comparable, well-defined result. Each measured implementation is compared with an independently computed reference matrix after every run.

## Q1: Matrix multiplication and memory access order

For `A[I x K]` and `B[K x J]`, four implementations were measured: the original `i-j-k` loop order (`matmul_ijk`), the cache-friendly `i-k-j` order (`matmul_ikj`), and versions that transpose A or B before multiplication. The complete 108-row result set is in `Lab1-GEMM-1/code/q1_results.csv`. Table 1 shows the square cases, which make the main trend easy to see.

| Algorithm | 256 x 256 x 256 (s) | 512 x 512 x 512 (s) | 1024 x 1024 x 1024 (s) |
| --- | ---: | ---: | ---: |
| `matmul_ijk` | 0.039320 | 0.303563 | 2.614750 |
| `matmul_ikj` | 0.016263 | 0.114815 | 0.981076 |
| `matmul_AT` | 0.047429 | 0.373731 | 3.090940 |
| `matmul_BT` | 0.014347 | 0.115826 | 0.891964 |

**Discussion.** C/C++ stores a two-dimensional array in row-major order. In `i-j-k`, the innermost loop accesses `B[k][j]` down a column, so successive reads are separated by an entire row of B. This has poor spatial locality. In `i-k-j`, the inner loop instead walks across the contiguous rows `B[k][j]` and `C[i][j]`; `A[i][k]` is reused for the entire row update. Therefore it has a substantially higher cache hit rate. Transposing B makes the inner-k access in the original dot-product formulation contiguous, while transposing A does not improve the problematic B access. The measurements agree with this analysis: `matmul_ikj` and `matmul_BT` are consistently much faster than `matmul_ijk` and `matmul_AT`.

## Q2: im2col convolution

The required convolution has input shape `1 x 3 x 56 x 56`, 64 filters of shape `3 x 3 x 3`, stride 1, and zero padding. Its output has shape `64 x 54 x 54`.

The implementation first creates the im2col matrix. One output location corresponds to one row, and one flattened receptive field has `3 x 3 x 3 = 27` values:

```text
im2col matrix A:  2916 x 27
weight matrix B:    27 x 64
GEMM result C:    2916 x 64
```

Here, `2916 = 54 x 54` output positions. The code packs the 64 convolution filters into B, computes `C = A x B`, and interprets `C[position][output_channel]` as `output[output_channel][row][column]`. A direct seven-loop convolution is also implemented as the reference.

| Measurement | Average time (s) |
| --- | ---: |
| Naive convolution | 0.000477 |
| im2col transform | 0.000024 |
| GEMM | 0.000719 |
| im2col + GEMM | 0.000744 |

The maximum difference between the naive and im2col+GEMM outputs was `0.000000`. On this small CPU example, the handwritten GEMM plus the extra im2col copy is slower than the compiler-optimized direct convolution. This does not change their mathematical equivalence. The purpose of im2col is to express convolution as a GEMM problem, allowing production frameworks to call highly optimized BLAS/GPU matrix-multiplication kernels for larger workloads.

## Q3: Cache optimization of GEMM

The Q1 `matmul_ikj` implementation is used as the baseline because it already has contiguous row accesses. Three portable cache-oriented variants were implemented in `Lab1-GEMM-1/code/q3_matmul.cpp`.

1. **Tiling (`matmul_tiled`)**: Processes 64 x 64 blocks. The B and C blocks are repeatedly used while resident in cache. Two such `uint32_t` blocks occupy about 32 KiB, leaving room for A-row data in a typical L1 data cache.
2. **Tiling plus four-way loop unrolling (`matmul_tiled_unroll4`)**: Handles four adjacent j values per iteration. This removes some loop-control work and exposes independent operations.
3. **Tiling plus write-back caching (`matmul_tiled_writeback4`)**: Keeps four C accumulators in scalar registers while processing a K block, then writes each result back once. This reduces repeated C loads/stores in the innermost computation.

All results below are averaged over 32 runs. Speedup is `time(matmul_ikj) / time(variant)`; greater than 1 means the variant is faster.

| Algorithm | 256 x 256 x 256 (s) | Speedup | 512 x 512 x 512 (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `matmul_ikj` baseline | 0.012034 | 1.00x | 0.086286 | 1.00x |
| `matmul_tiled` | 0.011239 | 1.07x | 0.089295 | 0.97x |
| `matmul_tiled_unroll4` | 0.012023 | 1.00x | 0.094827 | 0.91x |
| `matmul_tiled_writeback4` | 0.002868 | 4.20x | 0.022260 | 3.88x |

**Discussion.** Tiling provides a small gain at 256 but is slightly slower at 512 with the chosen 64 x 64 block size. This shows that blocking is not automatically beneficial: the block size, cache hierarchy, and compiler optimizations all matter. Manual loop unrolling also does not improve this Clang `-O3` build; the compiler already performs aggressive low-level optimization, and the manual body can add register pressure and instruction-cache cost.

Write-back caching is the clearest improvement. The baseline repeatedly updates a C element once for every k. The write-back kernel loads four C values, accumulates an entire K block in registers, then stores the four values once. This reduces C-memory traffic and exploits temporal locality, producing a 4.20x speedup at 256 and 3.88x at 512.

## Reproducibility and submitted code

Run the Q1, Q2, and Q3 experiments from `Lab1-GEMM-1/code`:

```bash
./run_q1_benchmark.sh 32 q1_results.csv
./run_im2col_convolution.sh 32
./run_q3_benchmark.sh 32 q3_results.csv
```

Q3 defaults to 256 and 512 because these complete within the session runtime limit and match the scope used in the inherited Lab 1 reference. To add the 1024 square case, run:

```bash
SIZES="256 512 1024" ./run_q3_benchmark.sh 32 q3_results.csv
```

The submitted Q2 code is `im2col_convolution.cpp`; the submitted Q3 code is `q3_matmul.cpp`. Both programs compile with standard C++17 and do not require architecture-specific SIMD extensions.
