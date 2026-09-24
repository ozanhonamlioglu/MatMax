#pragma once

#include "matx.hpp"

namespace matx
{

  class ChainOps {
  public:
    // will trigger chained operations.
    Matrix exec();

    // each operation will allocate memory on GPU
    void add();
    void sub();
    void mul();
    void scale();
    void transpose();
  };

}
