#pragma once

#include <vector>

struct Matrix {
  std::vector<float> mtx;
  int row_length;
};

class Matx {
public:
  Matrix add(const Matrix& A, const Matrix& B) const;

private:
  void matrix_addition_check(const Matrix& A, const Matrix& B) const;
};