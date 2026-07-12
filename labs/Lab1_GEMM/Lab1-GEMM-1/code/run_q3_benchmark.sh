#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

repetitions="${1:-32}"
output_file="${2:-q3_results.csv}"
algorithms=(matmul_ikj matmul_tiled matmul_tiled_unroll4 matmul_tiled_writeback4)
read -r -a sizes <<< "${SIZES:-256 512 1024}"

g++ q3_matmul.cpp -o q3_matmul -std=c++17 -O3 -Wall
trap 'rm -f q3_matmul' EXIT

echo "Algorithm,I,J,K,avg_time_seconds" > "$output_file"

for size in "${sizes[@]}"; do
  for algorithm in "${algorithms[@]}"; do
    echo "Running $algorithm with A=${size}x${size}, B=${size}x${size}" >&2
    output="$(./q3_matmul "$algorithm" "$size" "$size" "$size" "$repetitions")"
    average="$(awk '/Avg Time for Calculation:/ { print $5 }' <<< "$output")"
    echo "$algorithm,$size,$size,$size,$average" >> "$output_file"
  done
done

echo "Wrote results to $output_file" >&2
