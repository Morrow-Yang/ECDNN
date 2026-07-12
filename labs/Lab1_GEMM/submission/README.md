# ECDNN Lab 1 Submission Manifest

## Files for Grading

| File | Corresponding item | What it demonstrates |
| --- | --- | --- |
| `Lab1_Report.pdf` | Q1--Q3 report | Experimental environment, methodology, measurements, and discussion for all questions. |
| `Q2_im2col_convolution.cpp` | Q2 | Direct convolution reference, im2col lowering, filter packing, GEMM, output mapping, timing, and numerical verification. |
| `Q3_cache_optimized_gemm.cpp` | Q3 | Baseline `ikj` GEMM and three cache-oriented variants: tiling, tiling plus loop unrolling, and tiling plus write-back caching. |
| `Q1_results.csv` | Q1 evidence | All 108 Q1 results: 27 independent `(I, J, K)` shapes over `{256, 512, 1024}` and four GEMM access patterns. |
| `Q3_results.csv` | Q3 evidence | 32-run average timings for the Q3 algorithms at `256^3` and `512^3`. |

## Q1 Evidence

Q1 compares four mathematically equivalent matrix-multiplication implementations:

```text
matmul_ijk  : original i-j-k loop order
matmul_ikj  : row-contiguous i-k-j loop order
matmul_AT   : transpose A before multiplication
matmul_BT   : transpose B before multiplication
```

`Q1_results.csv` contains the complete required experiment grid. The PDF presents the 256/512 cases in the supplied report-table style and explicitly reports the largest square 1024 case; the CSV provides all remaining 1024 and non-square measurements.

## Q2: Build and Expected Check

Compile and run with a C++17 compiler:

```bash
g++ Q2_im2col_convolution.cpp -std=c++17 -O3 -Wall -o q2
./q2 32
```

The program uses the required configuration:

```text
Input:  1 x 3 x 56 x 56
Kernel: 64 filters x 3 x 3 x 3
Output: 64 x 54 x 54
```

It prints an im2col matrix shape of `2916 x 27`, a weight matrix shape of `27 x 64`, and `Maximum difference: 0.000000` when compared with direct convolution.

## Q3: Build and Expected Check

Compile and run a selected implementation with:

```bash
g++ Q3_cache_optimized_gemm.cpp -std=c++17 -O3 -Wall -o q3
./q3 matmul_tiled_writeback4 512 512 512 32
```

Supported algorithm arguments are:

```text
matmul_ikj
matmul_tiled
matmul_tiled_unroll4
matmul_tiled_writeback4
```

Each run computes an independent reference result and uses an assertion to verify the optimized output. The report discusses why write-back caching reduces repeated C-matrix memory traffic and reports the measured speedups.

