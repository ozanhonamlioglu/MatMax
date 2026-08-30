#include "matx.hpp"

#include <iostream>

extern "C" void matx_add(float* A, float* B, float* Buffer, int N);

// PUBLIC
Matrix Matx::add(const Matrix& A, const Matrix& B) const {
  matrix_addition_check(A, B);

  Matrix result;
  result.row_length = A.row_length;
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

// PRIVATE
void Matx::matrix_addition_check(const Matrix& A, const Matrix& B) const {
  if(A.mtx.size() != B.mtx.size()) {
    throw std::invalid_argument("Matrices must have the same total elements.");
  }

  if (A.row_length != B.row_length) {
    throw std::invalid_argument("Matrix addition failed: Matrix shapes/row lengths do not match!");
  }
}