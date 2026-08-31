#pragma once

#include <vector>
#include <stdexcept>

struct Matrix {
  std::vector<float> mtx;
  int num_cols; // length of a single row

  int num_rows() const {
    return mtx.size() / num_cols;
  }

  float get_cell_at(int row_i, int col_i) const {
    if(row_i < 0 || col_i < 0) {
      throw std::out_of_range("get_cell_at: indices must be non-negative.");
    }

    if(row_i >= num_rows()) {
      throw std::out_of_range("get_cell_at: row index out of bounds.");
    }

    if(col_i >= num_cols) {
      throw std::out_of_range("get_cell_at: col index out of bounds.");
    }

    return mtx[row_i * num_cols + col_i];
  }

  // Matrix transpose() const {}; will be added
};

class Matx {
public:
  Matrix add(const Matrix& A, const Matrix& B) const;
  Matrix mul(const Matrix& A, const Matrix& B) const;

private:
  void matrix_addition_check(const Matrix& A, const Matrix& B) const;
  void matrix_multiplication_check(const Matrix& A, const Matrix& B) const;
};