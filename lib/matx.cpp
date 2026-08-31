#include "matx.hpp"
#include "cuda/matx_ops.cuh"

#include <iostream>
#include <stdexcept>

// PUBLIC
Matrix Matx::add(const Matrix& A, const Matrix& B) const {
  matrix_addition_check(A, B);

  Matrix result;
  result.num_cols = A.num_cols;
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

Matrix Matx::mul(const Matrix& A, const Matrix& B) const {

}

// PRIVATE
void Matx::matrix_addition_check(const Matrix& A, const Matrix& B) const {
  if(A.mtx.size() != B.mtx.size()) {
    throw std::invalid_argument("Matrices must have the same total elements.");
  }

  if (A.num_cols != B.num_cols) {
    throw std::invalid_argument("Matrix addition failed: Matrix shapes/row lengths do not match!");
  }
}

void Matx::matrix_multiplication_check(const Matrix& A, const Matrix& B) const {
  if (A.num_cols != B.num_rows()) {
    throw std::invalid_argument("Matrix multiplication failed: A's column count must match B's row count!");
  }
}