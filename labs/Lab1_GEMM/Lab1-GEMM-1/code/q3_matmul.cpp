// Build: g++ q3_matmul.cpp -o q3_matmul -std=c++17 -O3 -Wall
// Run:   ./q3_matmul <algorithm> <I> <K> <J> [repetitions]

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

constexpr int kBlockSize = 64;
constexpr int kUnrollFactor = 4;
using Matrix = std::vector<std::uint32_t>;

struct Problem {
  int I;
  int K;
  int J;
  Matrix A;
  Matrix B;
  Matrix C;
  Matrix reference;
};

double now() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::uint32_t& at(Matrix& matrix, int columns, int row, int column) {
  return matrix[row * columns + column];
}

std::uint32_t at(const Matrix& matrix, int columns, int row, int column) {
  return matrix[row * columns + column];
}

Problem init(int I, int K, int J) {
  Problem problem{I, K, J, Matrix(I * K), Matrix(K * J), Matrix(I * J), Matrix(I * J)};
  for (int i = 0; i < I; ++i) {
    for (int k = 0; k < K; ++k) at(problem.A, K, i, k) = rand();
  }
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) at(problem.B, J, k, j) = rand();
  }

  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      for (int k = 0; k < K; ++k) {
        at(problem.reference, J, i, j) += at(problem.A, K, i, k) * at(problem.B, J, k, j);
      }
    }
  }
  return problem;
}

void test(const Problem& problem) {
  assert(problem.C == problem.reference);
}

// Q1's cache-friendly baseline: B[k][j] and C[i][j] are traversed by rows.
void matmul_ikj(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int i = 0; i < problem.I; ++i) {
    for (int k = 0; k < problem.K; ++k) {
      const std::uint32_t a = at(problem.A, problem.K, i, k);
      for (int j = 0; j < problem.J; ++j) {
        at(problem.C, problem.J, i, j) += a * at(problem.B, problem.J, k, j);
      }
    }
  }
}

// Cache tiling: the B and C blocks are reused while they remain in cache.
void matmul_tiled(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int ii = 0; ii < problem.I; ii += kBlockSize) {
    const int i_end = std::min(ii + kBlockSize, problem.I);
    for (int kk = 0; kk < problem.K; kk += kBlockSize) {
      const int k_end = std::min(kk + kBlockSize, problem.K);
      for (int jj = 0; jj < problem.J; jj += kBlockSize) {
        const int j_end = std::min(jj + kBlockSize, problem.J);
        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            const std::uint32_t a = at(problem.A, problem.K, i, k);
            for (int j = jj; j < j_end; ++j) {
              at(problem.C, problem.J, i, j) += a * at(problem.B, problem.J, k, j);
            }
          }
        }
      }
    }
  }
}

// Tiling plus manual four-way unrolling of the contiguous j loop.
void matmul_tiled_unroll4(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int ii = 0; ii < problem.I; ii += kBlockSize) {
    const int i_end = std::min(ii + kBlockSize, problem.I);
    for (int kk = 0; kk < problem.K; kk += kBlockSize) {
      const int k_end = std::min(kk + kBlockSize, problem.K);
      for (int jj = 0; jj < problem.J; jj += kBlockSize) {
        const int j_end = std::min(jj + kBlockSize, problem.J);
        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            const std::uint32_t a = at(problem.A, problem.K, i, k);
            int j = jj;
            for (; j + kUnrollFactor <= j_end; j += kUnrollFactor) {
              at(problem.C, problem.J, i, j) += a * at(problem.B, problem.J, k, j);
              at(problem.C, problem.J, i, j + 1) += a * at(problem.B, problem.J, k, j + 1);
              at(problem.C, problem.J, i, j + 2) += a * at(problem.B, problem.J, k, j + 2);
              at(problem.C, problem.J, i, j + 3) += a * at(problem.B, problem.J, k, j + 3);
            }
            for (; j < j_end; ++j) {
              at(problem.C, problem.J, i, j) += a * at(problem.B, problem.J, k, j);
            }
          }
        }
      }
    }
  }
}

// Tiling plus write caching: four C values stay in scalar registers across a K block,
// then are written back once instead of being loaded and stored for every k.
void matmul_tiled_writeback4(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int ii = 0; ii < problem.I; ii += kBlockSize) {
    const int i_end = std::min(ii + kBlockSize, problem.I);
    for (int jj = 0; jj < problem.J; jj += kBlockSize) {
      const int j_end = std::min(jj + kBlockSize, problem.J);
      for (int kk = 0; kk < problem.K; kk += kBlockSize) {
        const int k_end = std::min(kk + kBlockSize, problem.K);
        for (int i = ii; i < i_end; ++i) {
          int j = jj;
          for (; j + kUnrollFactor <= j_end; j += kUnrollFactor) {
            std::uint32_t c0 = at(problem.C, problem.J, i, j);
            std::uint32_t c1 = at(problem.C, problem.J, i, j + 1);
            std::uint32_t c2 = at(problem.C, problem.J, i, j + 2);
            std::uint32_t c3 = at(problem.C, problem.J, i, j + 3);
            for (int k = kk; k < k_end; ++k) {
              const std::uint32_t a = at(problem.A, problem.K, i, k);
              c0 += a * at(problem.B, problem.J, k, j);
              c1 += a * at(problem.B, problem.J, k, j + 1);
              c2 += a * at(problem.B, problem.J, k, j + 2);
              c3 += a * at(problem.B, problem.J, k, j + 3);
            }
            at(problem.C, problem.J, i, j) = c0;
            at(problem.C, problem.J, i, j + 1) = c1;
            at(problem.C, problem.J, i, j + 2) = c2;
            at(problem.C, problem.J, i, j + 3) = c3;
          }
          for (; j < j_end; ++j) {
            std::uint32_t c = at(problem.C, problem.J, i, j);
            for (int k = kk; k < k_end; ++k) {
              c += at(problem.A, problem.K, i, k) * at(problem.B, problem.J, k, j);
            }
            at(problem.C, problem.J, i, j) = c;
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " <matmul_ikj|matmul_tiled|matmul_tiled_unroll4|matmul_tiled_writeback4> <I> <K> <J> [repetitions]\n";
    return 1;
  }

  const std::string algorithm = argv[1];
  const int I = std::atoi(argv[2]);
  const int K = std::atoi(argv[3]);
  const int J = std::atoi(argv[4]);
  const int repetitions = argc == 6 ? std::atoi(argv[5]) : 32;
  if (I <= 0 || K <= 0 || J <= 0 || repetitions <= 0) {
    std::cerr << "I, K, J, and repetitions must be positive integers.\n";
    return 1;
  }

  Problem problem = init(I, K, J);
  double total_time = 0.0;
  for (int run = 0; run < repetitions; ++run) {
    const double start = now();
    if (algorithm == "matmul_ikj") {
      matmul_ikj(problem);
    } else if (algorithm == "matmul_tiled") {
      matmul_tiled(problem);
    } else if (algorithm == "matmul_tiled_unroll4") {
      matmul_tiled_unroll4(problem);
    } else if (algorithm == "matmul_tiled_writeback4") {
      matmul_tiled_writeback4(problem);
    } else {
      std::cerr << "Unknown algorithm: " << algorithm << '\n';
      return 1;
    }
    total_time += now() - start;
    test(problem);
  }

  std::cout << "Algorithm: " << algorithm << '\n';
  std::cout << "I: " << I << ", K: " << K << ", J: " << J << '\n';
  std::cout << "Block size: " << kBlockSize << ", unroll factor: " << kUnrollFactor << '\n';
  std::cout << "Avg Time for Calculation: " << total_time / repetitions << " seconds\n";
  return 0;
}
