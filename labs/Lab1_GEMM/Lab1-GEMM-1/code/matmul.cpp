// Build: g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall
// Run:   ./matmul <algorithm> <I> <K> <J> [repetitions]

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Unsigned arithmetic gives every loop order the same defined wraparound result.
using Matrix = std::vector<std::uint32_t>;

struct Problem {
  int I;
  int K;
  int J;
  Matrix A;
  Matrix B;
  Matrix C;
  Matrix C_groundtruth;
};

double get_time() {
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
    for (int k = 0; k < K; ++k) {
      at(problem.A, K, i, k) = rand();
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
      at(problem.B, J, k, j) = rand();
    }
  }

  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      for (int k = 0; k < K; ++k) {
        at(problem.C_groundtruth, J, i, j) += at(problem.A, K, i, k) * at(problem.B, J, k, j);
      }
    }
  }
  return problem;
}

void test(const Problem& problem) {
  assert(problem.C == problem.C_groundtruth);
}

void matmul(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int i = 0; i < problem.I; ++i) {
    for (int j = 0; j < problem.J; ++j) {
      for (int k = 0; k < problem.K; ++k) {
        at(problem.C, problem.J, i, j) += at(problem.A, problem.K, i, k) * at(problem.B, problem.J, k, j);
      }
    }
  }
}

void matmul_ikj(Problem& problem) {
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int i = 0; i < problem.I; ++i) {
    for (int k = 0; k < problem.K; ++k) {
      for (int j = 0; j < problem.J; ++j) {
        at(problem.C, problem.J, i, j) += at(problem.A, problem.K, i, k) * at(problem.B, problem.J, k, j);
      }
    }
  }
}

void matmul_AT(Problem& problem) {
  Matrix AT(problem.K * problem.I);
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int i = 0; i < problem.I; ++i) {
    for (int k = 0; k < problem.K; ++k) {
      at(AT, problem.I, k, i) = at(problem.A, problem.K, i, k);
    }
  }
  for (int i = 0; i < problem.I; ++i) {
    for (int j = 0; j < problem.J; ++j) {
      for (int k = 0; k < problem.K; ++k) {
        at(problem.C, problem.J, i, j) += at(AT, problem.I, k, i) * at(problem.B, problem.J, k, j);
      }
    }
  }
}

void matmul_BT(Problem& problem) {
  Matrix BT(problem.J * problem.K);
  std::fill(problem.C.begin(), problem.C.end(), 0);
  for (int k = 0; k < problem.K; ++k) {
    for (int j = 0; j < problem.J; ++j) {
      at(BT, problem.K, j, k) = at(problem.B, problem.J, k, j);
    }
  }
  for (int i = 0; i < problem.I; ++i) {
    for (int j = 0; j < problem.J; ++j) {
      for (int k = 0; k < problem.K; ++k) {
        at(problem.C, problem.J, i, j) += at(problem.A, problem.K, i, k) * at(BT, problem.K, j, k);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5 || argc > 6) {
    std::cerr << "Usage: " << argv[0] << " <matmul|matmul_ikj|matmul_AT|matmul_BT> <I> <K> <J> [repetitions]\n";
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
    const double start = get_time();
    if (algorithm == "matmul") {
      matmul(problem);
    } else if (algorithm == "matmul_ikj") {
      matmul_ikj(problem);
    } else if (algorithm == "matmul_AT") {
      matmul_AT(problem);
    } else if (algorithm == "matmul_BT") {
      matmul_BT(problem);
    } else {
      std::cerr << "Unknown algorithm: " << algorithm << '\n';
      return 1;
    }
    const double elapsed = get_time() - start;
    test(problem);
    total_time += elapsed;
  }

  std::cout << "Algorithm: " << algorithm << '\n';
  std::cout << "I: " << I << ", K: " << K << ", J: " << J << '\n';
  std::cout << "Avg Time for Calculation: " << total_time / repetitions << " seconds\n";
  return 0;
}
