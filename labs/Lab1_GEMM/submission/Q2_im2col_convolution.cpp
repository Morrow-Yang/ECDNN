// Build: g++ im2col_convolution.cpp -o im2col_convolution -std=c++17 -O3 -Wall
// Run:   ./im2col_convolution [repetitions]

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

constexpr int kBatch = 1;
constexpr int kInputHeight = 56;
constexpr int kInputWidth = 56;
constexpr int kInputChannels = 3;
constexpr int kOutputChannels = 64;
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 0;

constexpr int kOutputHeight = (kInputHeight + 2 * kPadding - kKernelSize) / kStride + 1;
constexpr int kOutputWidth = (kInputWidth + 2 * kPadding - kKernelSize) / kStride + 1;
constexpr int kPatchSize = kInputChannels * kKernelSize * kKernelSize;
constexpr int kOutputPositions = kOutputHeight * kOutputWidth;

using Tensor = std::vector<float>;

double now() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int input_index(int channel, int row, int column) {
  return (channel * kInputHeight + row) * kInputWidth + column;
}

int weight_index(int output_channel, int input_channel, int kernel_row, int kernel_column) {
  return ((output_channel * kInputChannels + input_channel) * kKernelSize + kernel_row) * kKernelSize + kernel_column;
}

int patch_index(int input_channel, int kernel_row, int kernel_column) {
  return (input_channel * kKernelSize + kernel_row) * kKernelSize + kernel_column;
}

// Reference implementation: output is stored as [output_channel][output_position].
void naive_convolution(const Tensor& input, const Tensor& weights, Tensor& output) {
  std::fill(output.begin(), output.end(), 0.0f);
  for (int output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    for (int output_row = 0; output_row < kOutputHeight; ++output_row) {
      for (int output_column = 0; output_column < kOutputWidth; ++output_column) {
        const int output_position = output_row * kOutputWidth + output_column;
        float sum = 0.0f;
        for (int input_channel = 0; input_channel < kInputChannels; ++input_channel) {
          for (int kernel_row = 0; kernel_row < kKernelSize; ++kernel_row) {
            for (int kernel_column = 0; kernel_column < kKernelSize; ++kernel_column) {
              const int input_row = output_row * kStride + kernel_row - kPadding;
              const int input_column = output_column * kStride + kernel_column - kPadding;
              if (input_row >= 0 && input_row < kInputHeight && input_column >= 0 && input_column < kInputWidth) {
                sum += input[input_index(input_channel, input_row, input_column)] *
                       weights[weight_index(output_channel, input_channel, kernel_row, kernel_column)];
              }
            }
          }
        }
        output[output_channel * kOutputPositions + output_position] = sum;
      }
    }
  }
}

// Each row represents one output location's flattened C x R x S input patch.
void im2col(const Tensor& input, Tensor& columns) {
  for (int output_row = 0; output_row < kOutputHeight; ++output_row) {
    for (int output_column = 0; output_column < kOutputWidth; ++output_column) {
      const int output_position = output_row * kOutputWidth + output_column;
      for (int input_channel = 0; input_channel < kInputChannels; ++input_channel) {
        for (int kernel_row = 0; kernel_row < kKernelSize; ++kernel_row) {
          for (int kernel_column = 0; kernel_column < kKernelSize; ++kernel_column) {
            const int input_row = output_row * kStride + kernel_row - kPadding;
            const int input_column = output_column * kStride + kernel_column - kPadding;
            const int column = patch_index(input_channel, kernel_row, kernel_column);
            columns[output_position * kPatchSize + column] =
                (input_row >= 0 && input_row < kInputHeight && input_column >= 0 && input_column < kInputWidth)
                    ? input[input_index(input_channel, input_row, input_column)]
                    : 0.0f;
          }
        }
      }
    }
  }
}

// Arrange filters as matrix B with shape [patch_size][output_channels].
void pack_weights(const Tensor& weights, Tensor& packed_weights) {
  for (int output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    for (int input_channel = 0; input_channel < kInputChannels; ++input_channel) {
      for (int kernel_row = 0; kernel_row < kKernelSize; ++kernel_row) {
        for (int kernel_column = 0; kernel_column < kKernelSize; ++kernel_column) {
          const int row = patch_index(input_channel, kernel_row, kernel_column);
          packed_weights[row * kOutputChannels + output_channel] =
              weights[weight_index(output_channel, input_channel, kernel_row, kernel_column)];
        }
      }
    }
  }
}

// C[P x M] = A[P x K] * B[K x M], with P=output positions, K=patch size, M=output channels.
void gemm(const Tensor& columns, const Tensor& packed_weights, Tensor& output_matrix) {
  std::fill(output_matrix.begin(), output_matrix.end(), 0.0f);
  for (int output_position = 0; output_position < kOutputPositions; ++output_position) {
    for (int patch_value = 0; patch_value < kPatchSize; ++patch_value) {
      const float activation = columns[output_position * kPatchSize + patch_value];
      const float* weight_row = &packed_weights[patch_value * kOutputChannels];
      float* output_row = &output_matrix[output_position * kOutputChannels];
      for (int output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        output_row[output_channel] += activation * weight_row[output_channel];
      }
    }
  }
}

float max_difference(const Tensor& reference, const Tensor& output_matrix) {
  float maximum = 0.0f;
  for (int output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    for (int output_position = 0; output_position < kOutputPositions; ++output_position) {
      const float gemm_value = output_matrix[output_position * kOutputChannels + output_channel];
      const float difference = std::abs(reference[output_channel * kOutputPositions + output_position] - gemm_value);
      maximum = std::max(maximum, difference);
    }
  }
  return maximum;
}

int main(int argc, char* argv[]) {
  const int repetitions = argc == 2 ? std::max(1, std::atoi(argv[1])) : 20;
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [repetitions]\n";
    return 1;
  }

  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  Tensor input(kInputChannels * kInputHeight * kInputWidth);
  Tensor weights(kOutputChannels * kPatchSize);
  for (float& value : input) value = distribution(generator);
  for (float& value : weights) value = distribution(generator);

  Tensor naive_output(kOutputChannels * kOutputPositions);
  Tensor columns(kOutputPositions * kPatchSize);
  Tensor packed_weights(kPatchSize * kOutputChannels);
  Tensor output_matrix(kOutputPositions * kOutputChannels);

  pack_weights(weights, packed_weights);
  naive_convolution(input, weights, naive_output);
  im2col(input, columns);
  gemm(columns, packed_weights, output_matrix);
  const float error = max_difference(naive_output, output_matrix);
  assert(error < 1e-4f);

  double naive_total = 0.0;
  double im2col_total = 0.0;
  double gemm_total = 0.0;
  for (int run = 0; run < repetitions; ++run) {
    double start = now();
    naive_convolution(input, weights, naive_output);
    naive_total += now() - start;

    start = now();
    im2col(input, columns);
    im2col_total += now() - start;

    start = now();
    gemm(columns, packed_weights, output_matrix);
    gemm_total += now() - start;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Input:  " << kBatch << " x " << kInputChannels << " x " << kInputHeight << " x " << kInputWidth << '\n';
  std::cout << "Kernel: " << kOutputChannels << " filters x " << kInputChannels << " x " << kKernelSize << " x " << kKernelSize << '\n';
  std::cout << "Output: " << kOutputChannels << " x " << kOutputHeight << " x " << kOutputWidth << '\n';
  std::cout << "im2col matrix: " << kOutputPositions << " x " << kPatchSize << '\n';
  std::cout << "Weight matrix: " << kPatchSize << " x " << kOutputChannels << '\n';
  std::cout << "Maximum difference: " << error << '\n';
  std::cout << "Average naive convolution time: " << naive_total / repetitions << " seconds\n";
  std::cout << "Average im2col time: " << im2col_total / repetitions << " seconds\n";
  std::cout << "Average GEMM time: " << gemm_total / repetitions << " seconds\n";
  std::cout << "Average im2col + GEMM time: " << (im2col_total + gemm_total) / repetitions << " seconds\n";
  return 0;
}
