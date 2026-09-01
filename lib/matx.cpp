#include "matx.hpp"
#include "cuda/matx_ops.cuh"

#include <iostream>
#include <stdexcept>
#include <random>

// PUBLIC
Matrix Matx::add(const Matrix& A, const Matrix& B) const {
  matrix_elementwise_check(A, B);

  Matrix result;
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

Matrix Matx::sub(const Matrix& A, const Matrix& B) const {
  matrix_elementwise_check(A, B);

  Matrix result;
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

Matrix Matx::scale(const Matrix& A, float scalar) const {
  Matrix result;
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

Matrix Matx::mul(const Matrix& A, const Matrix& B) const {
  matrix_multiplication_check(A, B);

  int M = A.num_rows();
  int K = A.dims;
  int P = B.dims;

  Matrix result;
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

Matrix Matx::zeros(int h, int d) {
  Matrix mat;
  mat.dims = d;
  mat.mtx.resize(h * d);
  return mat;
}

Matrix Matx::random(int h, int d) {
  Matrix mat;
  mat.dims = d;
  mat.mtx.resize(h * d);

  static std::mt19937 gen{std::random_device{}()};
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for(float& cell : mat.mtx) {
    cell = dist(gen);
  }

  return mat;
}

// PRIVATE
void Matx::matrix_elementwise_check(const Matrix& A, const Matrix& B) const {
  if(A.mtx.size() != B.mtx.size()) {
    throw std::invalid_argument("Matrices must have the same total elements.");
  }

  if (A.dims != B.dims) {
    throw std::invalid_argument("Elementwise op failed: Matrix shapes/row lengths do not match!");
  }
}

void Matx::matrix_multiplication_check(const Matrix& A, const Matrix& B) const {
  if (A.dims != B.num_rows()) {
    throw std::invalid_argument("Matrix multiplication failed: A's column count must match B's row count!");
  }
}