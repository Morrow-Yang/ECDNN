# Lab 1: General Matrix Multiplication

This directory contains the current Lab 1 materials and solutions for **Efficient Computing of Deep Neural Networks**. The lab follows one performance chain:

```text
matrix access order -> cache locality -> GEMM optimization -> im2col convolution
```

## Submission files

Submit the following files to the course system after replacing the report's name and student-ID placeholders:

1. `Lab1_Report.pdf`
2. `Lab1-GEMM-1/code/im2col_convolution.cpp` (Q2)
3. `Lab1-GEMM-1/code/q3_matmul.cpp` (Q3)

`q1_results.csv` and `q3_results.csv` are useful supporting evidence. Upload them too if the system permits extra files; otherwise, keep them locally as the source data for the tables in the report.

## File map

| File | Question / role | Purpose |
| --- | --- | --- |
| `Lab1-report.pdf` | Assignment brief | Official Lab 1 requirements. |
| `lab1_slides.pdf` | Background | Cache hierarchy, matrix multiplication, transpose, and loop-order examples. |
| `format-report.pdf` | Report reference | Example layout for the Q1 timing table. |
| `Lab1_Report.pdf` | Final report | Ready-to-submit report. Fill in name and student ID first. |
| `Lab1_Report.docx` | Final report | Editable Word version of the report. |
| `Lab1_Report.md` | Final report source | Editable Markdown source used to generate the Word and PDF reports. |
| `convolution_playground.html` | Study aid | Interactive visual explanation of direct convolution, im2col, and MEC. Open it in a browser. |
| `Lab1-GEMM-1/code/matmul.cpp` | Q1 | Configurable baseline GEMM implementations: `ijk`, `ikj`, transpose A, and transpose B. |
| `Lab1-GEMM-1/code/run_q1_benchmark.sh` | Q1 | Runs the four Q1 algorithms for all 27 combinations of I, J, and K. |
| `Lab1-GEMM-1/code/q1_results.csv` | Q1 evidence | Measured results for all 108 Q1 experiments. |
| `Lab1-GEMM-1/code/im2col_convolution.cpp` | Q2 | Naive convolution reference plus im2col, weight packing, GEMM, output mapping, and correctness verification. |
| `Lab1-GEMM-1/code/run_im2col_convolution.sh` | Q2 | Builds and runs the Q2 implementation. |
| `Lab1-GEMM-1/code/q3_matmul.cpp` | Q3 | Cache-optimization benchmark: tiling, loop unrolling, and write-back caching. |
| `Lab1-GEMM-1/code/run_q3_benchmark.sh` | Q3 | Builds Q3 and records its formal timing results. |
| `Lab1-GEMM-1/code/q3_results.csv` | Q3 evidence | 32-run averages for each Q3 algorithm at 256 and 512 square matrices. |

## Q1: Matrix access order

Q1 measures four mathematically equivalent ways to compute `C = A x B`:

```text
matmul_ijk  : i-j-k loop order; B is accessed by columns in the inner loop
matmul_ikj  : i-k-j loop order; B and C are accessed by contiguous rows
matmul_AT   : transpose A before multiplication
matmul_BT   : transpose B before multiplication
```

Run all Q1 experiments:

```bash
cd Lab1-GEMM-1/code
./run_q1_benchmark.sh 32 q1_results.csv
```

## Q2: im2col convolution

Q2 implements a standard convolution layer with the required fixed parameters:

```text
Input:  1 x 3 x 56 x 56
Kernel: 64 filters x 3 x 3 x 3
Output: 64 x 54 x 54
```

The code lowers convolution to matrix multiplication:

```text
im2col matrix: 2916 x 27
weight matrix:   27 x 64
GEMM result:   2916 x 64
```

Run Q2:

```bash
cd Lab1-GEMM-1/code
./run_im2col_convolution.sh 32
```

The program compares im2col+GEMM with a direct convolution reference and asserts that their outputs match.

## Q3: Cache optimization

Q3 uses `matmul_ikj` as the cache-friendly baseline and compares three optimizations:

```text
matmul_tiled             : 64 x 64 cache tiling
matmul_tiled_unroll4     : tiling plus four-way unrolling of the j loop
matmul_tiled_writeback4  : tiling plus four C accumulators held before write-back
```

Run the formal Q3 experiment (256 and 512, 32 repetitions):

```bash
cd Lab1-GEMM-1/code
./run_q3_benchmark.sh 32 q3_results.csv
```

To include the 1024 square case as well:

```bash
SIZES="256 512 1024" ./run_q3_benchmark.sh 32 q3_results.csv
```

## Related course concepts

- `lab1_slides.pdf`: cache hierarchy, row-major layout, cache hit ratio, loop order, and transpose.
- `slides/Im1-GEMM.pdf` in the repository root: convolution, im2col, GEMM, and tensor memory layout.
- Q1 explains why access order changes GEMM performance.
- Q2 shows why convolution is commonly transformed into GEMM.
- Q3 applies cache-locality ideas to the GEMM computation created by Q2.
