#include "matx.hpp"
#include "cuda/matx_ops.cuh"

#include <iostream>
#include <stdexcept>
#include <random>

// PUBLIC
matx::Matrix matx::Ops::add(const matx::Matrix& A, const matx::Matrix& B) {
  matrix_elementwise_check(A, B);

  matx::Matrix result;
  result.dims = A.dims;
  result.mtx.resize(A.mtx.size());

  int N = static_cast<int>(A.mtx.size());

  matx_add(
    const_cast<float*>(A.mtx.data()),
    const_cast<float*>(B.mtx.data()),
    result.mtx.data(),
    N
  );

  return result;
}

matx::Matrix matx::Ops::sub(const matx::Matrix& A, const matx::Matrix& B) {
  matrix_elementwise_check(A, B);

  matx::Matrix result;
  result.dims = A.dims;
  result.mtx.resize(A.mtx.size());

  int N = static_cast<int>(A.mtx.size());

  matx_sub(
    const_cast<float*>(A.mtx.data()),
    const_cast<float*>(B.mtx.data()),
    result.mtx.data(),
    N
  );

  return result;
}

matx::Matrix matx::Ops::scale(const matx::Matrix& A, float scalar) {
  matx::Matrix result;
  result.dims = A.dims;
  result.mtx.resize(A.mtx.size());

  int N = static_cast<int>(A.mtx.size());

  matx_scale(
    const_cast<float*>(A.mtx.data()),
    result.mtx.data(),
    scalar,
    N
  );

  return result;
}

matx::Matrix matx::Ops::mul(const matx::Matrix& A, const matx::Matrix& B) {
  matrix_multiplication_check(A, B);

  int M = A.num_rows();
  int K = A.dims;
  int P = B.dims;

  matx::Matrix result;
  result.dims = P;
  result.mtx.resize(M * P);

  matx_mul(
    const_cast<float*>(A.mtx.data()),
    const_cast<float*>(B.mtx.data()),
    result.mtx.data(),
    M, K, P
  );

  return result;
}

matx::Matrix matx::Ops::zeros(int h, int d) {
  matx::Matrix mat;
  mat.dims = d;
  mat.mtx.resize(h * d);
  return mat;
}

matx::Matrix matx::Ops::randomf(int h, int d) {
  matx::Matrix mat;
  mat.dims = d;
  mat.mtx.resize(h * d);
  matx_randomf(h * d, mat.mtx.data());
  return mat;
}

// PRIVATE
void matx::Ops::matrix_elementwise_check(const matx::Matrix& A, const matx::Matrix& B) {
  if(A.mtx.size() != B.mtx.size()) {
    throw std::invalid_argument("Matrices must have the same total elements.");
  }

  if (A.dims != B.dims) {
    throw std::invalid_argument("Elementwise op failed: Matrix shapes/row lengths do not match!");
  }
}

void matx::Ops::matrix_multiplication_check(const matx::Matrix& A, const matx::Matrix& B) {
  if (A.dims != B.num_rows()) {
    throw std::invalid_argument("Matrix multiplication failed: A's column count must match B's row count!");
  }
}