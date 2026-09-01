#include "lib/matx.hpp"

#include <iostream>
#include <vector>

#define OK "OK"
#define FAIL "FAILED"

void test_float(float x, float y) {
  if(x == y) {
    std::cout << OK << std::endl;
  } else {
    std::cout << FAIL << std::endl;
  }
}

void test_mat_add() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .dims = 3 };
  Matrix B { .mtx = _B, .dims = 3 };

  Matx matx;
  Matrix result = matx.add(A, B);

  float cell_value = result.get_cell_at(0, 2);
  test_float(cell_value, 6.0f);
}

void test_mat_mul() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .dims = 2 };
  Matrix B { .mtx = _B, .dims = 3 };

  Matx matx;
  Matrix result = matx.mul(A, B);

  float cell_value_1 = result.get_cell_at(0, 0);
  float cell_value_2 = result.get_cell_at(2, 1);
  test_float(cell_value_1, 9.0f);
  test_float(cell_value_2, 40.0f);
}

void test_mat_mul_transpose() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .dims = 3 };

  // Before transpose: A is 2x3
  float before_value = A.get_cell_at(1, 0);
  test_float(before_value, 4.0f);

  Matrix trans = A.transpose();

  // After transpose: trans is 3x2, trans(col, row) == A(row, col)
  float after_value = trans.get_cell_at(0, 1);
  test_float(after_value, 4.0f);
}

void test_run_all() {
  test_mat_add();
  test_mat_mul();
  test_mat_mul_transpose();
}