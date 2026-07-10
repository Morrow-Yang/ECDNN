#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

repetitions="${1:-32}"
output_file="${2:-q1_results.csv}"
algorithms=(matmul matmul_ikj matmul_AT matmul_BT)
read -r -a sizes <<< "${SIZES:-256 512 1024}"

g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall
trap 'rm -f matmul' EXIT

echo "Algorithm,I,J,K,avg_time_seconds" > "$output_file"

for I in "${sizes[@]}"; do
  for K in "${sizes[@]}"; do
    for J in "${sizes[@]}"; do
      for algorithm in "${algorithms[@]}"; do
        echo "Running $algorithm with A=${I}x${K}, B=${K}x${J}" >&2
        output="$(./matmul "$algorithm" "$I" "$K" "$J" "$repetitions")"
        average="$(awk '/Avg Time for Calculation:/ { print $5 }' <<< "$output")"
        report_algorithm="$algorithm"
        if [[ "$algorithm" == "matmul" ]]; then
          report_algorithm="matmul_ijk"
        fi
        echo "$report_algorithm,$I,$J,$K,$average" >> "$output_file"
      done
    done
  done
done

echo "Wrote results to $output_file" >&2
