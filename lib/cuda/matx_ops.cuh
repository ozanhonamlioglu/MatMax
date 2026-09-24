#pragma once

#include <chrono>

void matx_add(float* A, float* B, float* Buffer, int N);

void matx_sub(float* A, float* B, float* Buffer, int N);

void matx_scale(float* A, float* Buffer, float scalar, int N);

void matx_mul(float* A, float* B, float* Buffer, int M, int K, int P);

bool matx_equal(float* A, float* B, int length);

void matx_randomf(
  int length, 
  float* out, 
  unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count()
);

void matx_transpose(float* A, int Row, int Col, float* Buffer);