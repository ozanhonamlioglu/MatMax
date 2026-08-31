#include "lib/matx.hpp"

#include <iostream>
#include <vector>

void test_mat_add() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .num_cols = 3 };
  Matrix B { .mtx = _B, .num_cols = 3 };

  Matx matx;
  Matrix result = matx.add(A, B);

  // Print elements separated by a space
  for (const auto& element : result.mtx) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

void test_mat_mul() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  Matrix A { .mtx = _A, .num_cols = 3 };
  Matrix B { .mtx = _B, .num_cols = 3 };

  Matx matx;
  Matrix result = matx.mul(A, B);

  // Print elements separated by a space
  for (const auto& element : result.mtx) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

void test_run_all() {
  test_mat_add();
  test_mat_mul();
}