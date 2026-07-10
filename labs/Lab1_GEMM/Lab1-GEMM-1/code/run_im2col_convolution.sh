#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
g++ im2col_convolution.cpp -o im2col_convolution -std=c++17 -O3 -Wall
./im2col_convolution "${1:-20}"
rm -f im2col_convolution
