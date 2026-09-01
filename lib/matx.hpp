#pragma once

#include <vector>
#include <stdexcept>

struct Matrix {
  std::vector<float> mtx;
  int dims; // length of a single row

  int num_rows() const {
    return mtx.size() / dims;
  }

  float get_cell_at(int row_i, int col_i) const {
    if(row_i < 0 || col_i < 0) {
      throw std::out_of_range("get_cell_at: indices must be non-negative.");
    }

    if(row_i >= num_rows()) {
      throw std::out_of_range("get_cell_at: row index out of bounds.");
    }

    if(col_i >= dims) {
      throw std::out_of_range("get_cell_at: col index out of bounds.");
    }

    return mtx[row_i * dims + col_i];
  }

  Matrix transpose() const {
    int rows = num_rows();

    Matrix trans;
    trans.dims = rows;
    trans.mtx.resize(mtx.size());

    for(int row_i = 0; row_i < rows; ++row_i) {
      for(int col_i = 0; col_i < dims; ++col_i) {
        trans.mtx[col_i * rows + row_i] = get_cell_at(row_i, col_i);
      }
    }

    return trans;
  }
};

class Matx {
public:
  static Matrix zeros(int h, int d);

  Matrix add(const Matrix& A, const Matrix& B) const;
  Matrix mul(const Matrix& A, const Matrix& B) const;

private:
  void matrix_addition_check(const Matrix& A, const Matrix& B) const;
  void matrix_multiplication_check(const Matrix& A, const Matrix& B) const;
};