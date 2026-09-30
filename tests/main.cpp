#include "utils.hpp"
#include "lib/matx.hpp"
#include "lib/chain.hpp"

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <concepts>
#include <source_location>

void test_mat_add() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  matx::Matrix A { .mtx = _A, .dims = 3 };
  matx::Matrix B { .mtx = _B, .dims = 3 };

  matx::Matrix result = matx::Ops::add(A, B);

  float cell_value = result.get_cell_at(0, 2);
  test_compare(cell_value, 6.0f);
}

void test_mat_mul() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 2, 3, 4, 5, 6};

  matx::Matrix A { .mtx = _A, .dims = 2 };
  matx::Matrix B { .mtx = _B, .dims = 3 };

  matx::Matrix result = matx::Ops::mul(A, B);

  float cell_value_1 = result.get_cell_at(0, 0);
  float cell_value_2 = result.get_cell_at(2, 1);
  test_compare(cell_value_1, 9.0f);
  test_compare(cell_value_2, 40.0f);
}

void test_mat_transpose_host() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};

  matx::Matrix A { .mtx = _A, .dims = 3 };

  // Before transpose: A is 2x3
  float before_value = A.get_cell_at(1, 0);
  test_compare(before_value, 4.0f);

  matx::Matrix trans = A.transpose();

  // After transpose: trans is 3x2, trans(col, row) == A(row, col)
  float after_value = trans.get_cell_at(0, 1);
  test_compare(after_value, 4.0f);
}

void test_mat_transpose_device() {
  // Non-square so a swapped Row/Col would be caught
  matx::Matrix A = matx::Ops::randomf(300, 170);

  matx::Matrix host_trans = A.transpose();
  matx::Matrix device_trans = matx::Ops::transpose(A);

  test_compare(device_trans.dims, host_trans.dims);
  test_compare(device_trans.num_rows(), host_trans.num_rows());
  test_compare(static_cast<int>(matx::Ops::is_equal(host_trans, device_trans)), 1);
}

void test_mat_sub() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};
  std::vector<float> _B = {1, 1, 1, 1, 1, 1};

  matx::Matrix A { .mtx = _A, .dims = 3 };
  matx::Matrix B { .mtx = _B, .dims = 3 };

  matx::Matrix result = matx::Ops::sub(A, B);

  float cell_value = result.get_cell_at(1, 2);
  test_compare(cell_value, 5.0f);
}

void test_mat_scale() {
  std::vector<float> _A = {1, 2, 3, 4, 5, 6};

  matx::Matrix A { .mtx = _A, .dims = 3 };

  matx::Matrix result = matx::Ops::scale(A, 2.0f);

  float cell_value = result.get_cell_at(1, 2);
  test_compare(cell_value, 12.0f);
}

void zeros_test() {
  matx::Matrix A = matx::Ops::zeros(5,10);
  test_compare(static_cast<int>(A.mtx.size()), 50);
}

void test_random() {
  matx::Matrix A = matx::Ops::randomf(5, 10);
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

void test_is_equal() {
  matx::Matrix A = matx::Ops::randomf(200, 200);
  matx::Matrix B = A;
  matx::Matrix C = matx::Ops::randomf(200, 200);

  test_compare(static_cast<int>(matx::Ops::is_equal(A, B)), 1);
  test_compare(static_cast<int>(matx::Ops::is_equal(A, C)), 0);
}

void chain_of_ops() {
  matx::Matrix A = matx::Ops::randomf(300, 170);
  matx::Matrix B = matx::Ops::randomf(300, 170);
  matx::Matrix C = matx::Ops::randomf(300, 170);

  // Regular ops: ((A + B) - C) * 2.5, each step round-trips through the host
  matx::Matrix expected = matx::Ops::scale(matx::Ops::sub(matx::Ops::add(A, B), C), 2.5f);

  // Chained ops: same computation, intermediates stay on the device
  matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };
  matx::ChainMatrix cB { .mtx = B.mtx.data(), .device = nullptr, .dims = B.dims, .rows = B.num_rows() };
  matx::ChainMatrix cC { .mtx = C.mtx.data(), .device = nullptr, .dims = C.dims, .rows = C.num_rows() };

  matx::ChainOps chain;
  matx::ChainMatrix sum = chain.add(cA, cB);
  matx::ChainMatrix diff = chain.sub(sum, cC);
  chain.scale(diff, 2.5f);
  matx::Matrix result = chain.complete();

  test_compare(result.dims, expected.dims);
  test_compare(result.num_rows(), expected.num_rows());
  test_compare(static_cast<int>(matx::Ops::is_equal(result, expected)), 1);
}

void chain_without_complete() {
  matx::Matrix A = matx::Ops::randomf(50, 40);
  matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };

  // complete() is never called, ~ChainOps must free the device buffers
  {
    matx::ChainOps chain;
    matx::ChainMatrix sum = chain.add(cA, cA);
    chain.scale(sum, 2.0f);
  }

  test_compare(1, 1);
}

void chain_shape_mismatch() {
  matx::Matrix A = matx::Ops::randomf(4, 6);
  matx::Matrix B = matx::Ops::randomf(6, 4); // same element count, different shape
  matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };
  matx::ChainMatrix cB { .mtx = B.mtx.data(), .device = nullptr, .dims = B.dims, .rows = B.num_rows() };

  matx::ChainOps chain;
  bool thrown = false;
  try {
    chain.add(cA, cB);
  } catch(const std::invalid_argument&) {
    thrown = true;
  }

  test_compare(static_cast<int>(thrown), 1);
}

void chain_mul_transpose() {
  // Non-square everywhere so a swapped dims/rows would be caught
  matx::Matrix A = matx::Ops::randomf(300, 170);
  matx::Matrix B = matx::Ops::randomf(300, 120);

  // Regular ops: transpose(A) * B -> 170 x 120, then transpose -> 120 x 170
  matx::Matrix expected = matx::Ops::transpose(matx::Ops::mul(matx::Ops::transpose(A), B));

  matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };
  matx::ChainMatrix cB { .mtx = B.mtx.data(), .device = nullptr, .dims = B.dims, .rows = B.num_rows() };

  matx::ChainOps chain;
  matx::ChainMatrix tA = chain.transpose(cA);
  matx::ChainMatrix prod = chain.mul(tA, cB);
  chain.transpose(prod);
  matx::Matrix result = chain.complete();

  test_compare(result.dims, expected.dims);
  test_compare(result.num_rows(), expected.num_rows());
  test_compare(static_cast<int>(matx::Ops::is_equal(result, expected)), 1);
}

void chain_mul_shape_mismatch() {
  matx::Matrix A = matx::Ops::randomf(4, 6);
  matx::Matrix B = matx::Ops::randomf(4, 6); // A's 6 columns != B's 4 rows
  matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };
  matx::ChainMatrix cB { .mtx = B.mtx.data(), .device = nullptr, .dims = B.dims, .rows = B.num_rows() };

  matx::ChainOps chain;
  bool thrown = false;
  try {
    chain.mul(cA, cB);
  } catch(const std::invalid_argument&) {
    thrown = true;
  }

  test_compare(static_cast<int>(thrown), 1);
}

int main(int argc, char **argv) {
  test_mat_add();
  test_mat_mul();
  test_mat_transpose_host();
  test_mat_transpose_device();
  test_mat_sub();
  test_mat_scale();
  zeros_test();
  test_random();
  test_is_equal();
  chain_of_ops();
  chain_without_complete();
  chain_shape_mismatch();
  chain_mul_transpose();
  chain_mul_shape_mismatch();

  return 0;
}