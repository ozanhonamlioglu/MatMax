#pragma once

#include "matx.hpp"
#include <vector>

namespace matx
{

  enum class OpType {
    ADD = 1,
    SUB,
    MUL,
    SCALE,
    TRANSPOSE
  };

  struct ChainMatrix {
    float *mtx; // either mtx or device is null but user not need to worry for it
    float *device; // device pointer
    int dims, rows;
  };

  struct DeviceMemCollector {
    float *buffer;
    int dims, rows;
    DeviceMemCollector* next;
  };

  class ChainOps {  
  private:
    std::vector<DeviceMemCollector> collector;
    void update_list(float *buffer, int dims, int rows);
    static void matrix_elementwise_check(const ChainMatrix& A, const ChainMatrix& B);
    static void matrix_multiplication_check(const ChainMatrix& A, const ChainMatrix& B);

  public:
    ChainOps() = default;
    ~ChainOps(); // frees any device memory left if complete() was never called

    // the collector owns raw device pointers, a copy would free them twice
    ChainOps(const ChainOps&) = delete;
    ChainOps& operator=(const ChainOps&) = delete;

    // will trigger chained operations.
    Matrix complete();

    // each operation will allocate memory on GPU
    ChainMatrix add(const ChainMatrix& A, const ChainMatrix& B);
    ChainMatrix sub(const ChainMatrix& A, const ChainMatrix& B);
    ChainMatrix mul(const ChainMatrix& A, const ChainMatrix& B);
    ChainMatrix scale(const ChainMatrix& A, float scalar);
    ChainMatrix transpose(const ChainMatrix& A);
  };

}

/**
 * Each operation executes the command but doesn't move from device to host
 * Until "end" is called.
 * 
 * Each operation run on when executed but will not be freed or moved data to the host until end runs! Continue...
 */