#include "chain.hpp"
#include "cuda/matx_chain_ops.cuh"
#include <stdexcept>

void matx::ChainOps::update_list(float *buffer, int dims, int rows) {
  collector.push_back({buffer, dims, rows, nullptr});
}

matx::ChainOps::~ChainOps() {
  matx_chain_free(collector);
}

matx::Matrix matx::ChainOps::complete() {
  if(collector.empty()) {
    throw std::runtime_error("complete: no operations in the chain.");
  }

  const matx::DeviceMemCollector& last = collector.back();

  matx::Matrix result;
  result.dims = last.dims;
  result.mtx.resize(last.dims * last.rows);

  matx_chain_complete(collector, result.mtx.data());
  collector.clear();

  return result;
}

matx::ChainMatrix matx::ChainOps::add(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  matx::ChainMatrix buffer{};
  buffer.dims = A.dims;
  buffer.rows = A.rows;

  matrix_elementwise_check(A, B);
  float* deviceBuffer = matx_chain_add(A, B);

  update_list(deviceBuffer, buffer.dims, buffer.rows);
  buffer.device = deviceBuffer;
  return buffer;
}

matx::ChainMatrix matx::ChainOps::sub(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  matx::ChainMatrix buffer{};
  buffer.dims = A.dims;
  buffer.rows = A.rows;

  matrix_elementwise_check(A, B);
  float* deviceBuffer = matx_chain_sub(A, B);

  update_list(deviceBuffer, buffer.dims, buffer.rows);
  buffer.device = deviceBuffer;
  return buffer;
}

matx::ChainMatrix matx::ChainOps::scale(const matx::ChainMatrix& A, float scalar) {
  matx::ChainMatrix buffer{};
  buffer.dims = A.dims;
  buffer.rows = A.rows;

  float* deviceBuffer = matx_chain_scale(A, scalar);

  update_list(deviceBuffer, buffer.dims, buffer.rows);
  buffer.device = deviceBuffer;
  return buffer;
}

matx::ChainMatrix matx::ChainOps::mul(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  matx::ChainMatrix buffer{};
  buffer.dims = B.dims;
  buffer.rows = A.rows;

  matrix_multiplication_check(A, B);
  float* deviceBuffer = matx_chain_mul(A, B);

  update_list(deviceBuffer, buffer.dims, buffer.rows);
  buffer.device = deviceBuffer;
  return buffer;
}

matx::ChainMatrix matx::ChainOps::transpose(const matx::ChainMatrix& A) {
  matx::ChainMatrix buffer{};
  buffer.dims = A.rows;
  buffer.rows = A.dims;

  float* deviceBuffer = matx_chain_transpose(A);

  update_list(deviceBuffer, buffer.dims, buffer.rows);
  buffer.device = deviceBuffer;
  return buffer;
}

// PRIVATE
void matx::ChainOps::matrix_elementwise_check(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  if(A.dims * A.rows != B.dims * B.rows) {
    throw std::invalid_argument("Matrices must have the same total elements.");
  }

  if (A.dims != B.dims) {
    throw std::invalid_argument("Elementwise op failed: Matrix shapes/row lengths do not match!");
  }
}

void matx::ChainOps::matrix_multiplication_check(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  if (A.dims != B.rows) {
    throw std::invalid_argument("Matrix multiplication failed: A's column count must match B's row count!");
  }
}
