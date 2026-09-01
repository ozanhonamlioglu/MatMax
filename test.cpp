#include "lib/matx.hpp"

#include <iostream>
#include <vector>
#include <type_traits>
#include <concepts>
#include <source_location>

#define OK "OK"
#define FAIL "FAILED"

template <typename T>
concept IntOrFloat = std::same_as<T, int> || std::same_as<T, float>;

void test_compare(
  IntOrFloat auto x, 
  IntOrFloat auto y,
  const std::source_location& loc = std::source_location::current()) {
  std::cout << loc.function_name() << " -> ";

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
  test_compare(cell_value, 6.0f);
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
  test_compare(cell_value_1, 9.0f);
  test_compare(cell_value_2, 40.0f);
}

void test_mat_mul_transpose() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .dims = 3 };

  // Before transpose: A is 2x3
  float before_value = A.get_cell_at(1, 0);
  test_compare(before_value, 4.0f);

  Matrix trans = A.transpose();

  // After transpose: trans is 3x2, trans(col, row) == A(row, col)
  float after_value = trans.get_cell_at(0, 1);
  test_compare(after_value, 4.0f);
}

void test_mat_sub() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 1, 1, 1, 1, 1};

  Matrix A { .mtx = _A, .dims = 3 };
  Matrix B { .mtx = _B, .dims = 3 };

  Matx matx;
  Matrix result = matx.sub(A, B);

  float cell_value = result.get_cell_at(1, 2);
  test_compare(cell_value, 5.0f);
}

void test_mat_scale() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .dims = 3 };

  Matx matx;
  Matrix result = matx.scale(A, 2.0f);

  float cell_value = result.get_cell_at(1, 2);
  test_compare(cell_value, 12.0f);
}

void zeros_test() {
  Matrix A = Matx::zeros(5,10);
  test_compare(static_cast<int>(A.mtx.size()), 50);
}

void random_test() {
  Matrix A = Matx::random(5, 10);
  test_compare(static_cast<int>(A.mtx.size()), 50);
  test_compare(A.dims, 10);

  bool in_range = true;
  for(float cell : A.mtx) {
    if(cell < 0.0f || cell >= 1.0f) {
      in_range = false;
      break;
    }
  }

  test_compare(static_cast<int>(in_range), 1);
}

void test_run_all() {
  test_mat_add();
  test_mat_mul();
  test_mat_mul_transpose();
  test_mat_sub();
  test_mat_scale();
  zeros_test();
  random_test();
}