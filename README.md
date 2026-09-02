# MatMax

A CUDA-accelerated matrix library for C++20, implemented without external linear algebra dependencies such as cuBLAS or Eigen. Core operations (matrix multiplication, transpose, elementwise addition/subtraction, scalar scaling) are each implemented as CUDA kernels. A logistic regression implementation is included as an example application built entirely on top of the library's matrix operations.

## Purpose

The goal of this project is to provide GPU-accelerated linear algebra primitives implemented from first principles, and to demonstrate that a real machine learning algorithm — forward pass, loss, gradient computation, and weight update — can be expressed entirely in terms of those primitives. The library is intended as a foundation that can be extended with additional operations and used as the basis for further work.

## Project structure

```
matmax/
├── CMakeLists.txt
├── lib/
│   ├── matx.hpp                 # Matrix struct + Matx op-dispatch class
│   ├── matx.cpp                 # Matx method implementations (host side)
│   └── cuda/
│       ├── matx_ops.cuh         # CUDA kernel declarations
│       └── matx_ops.cu          # CUDA kernels + host launch wrappers
├── tests/
│   ├── main.cpp                  # unit tests for the matrix library
│   └── utils.hpp                 # test_compare helper
└── examples/
    └── ml/
        └── logistic-regression.cpp  # logistic regression example
```

Two executables come out of the build: `tests` (the library's unit tests) and `logreg` (the logistic regression example).

## Requirements

- CMake ≥ 3.18
- An NVIDIA GPU + driver
- CUDA Toolkit (developed against CUDA 12.4)
- A C++20-capable host compiler (GCC/Clang)

## Build & run

```bash
mkdir -p build && cd build
cmake ..
cmake --build .

./tests    # runs the matrix library test suite
./logreg   # trains & validates logistic regression
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so it targets whatever GPU is in the machine doing the build.

## The matrix library

### `Matrix` (`lib/matx.hpp`)

A row-major matrix: a flat `std::vector<float>` plus `dims`, the length of one row.

| Member | Description |
|---|---|
| `num_rows()` | `mtx.size() / dims` |
| `get_cell_at(row, col)` | Bounds-checked element access |
| `transpose()` | Returns a new transposed matrix (host-side index remapping) |

### `Matx` — the CUDA-backed operations

Every operation below allocates device memory, copies operands to the GPU, launches a kernel, copies the result back, and frees device memory. Kernels use a flat 1D launch (`threadIdx.x + blockIdx.x * blockDim.x`), 256 threads per block, with a bounds check so sizes that aren't clean multiples of the block size don't read/write out of range.

| Method | Shape rule | What it does |
|---|---|---|
| `Matx::zeros(h, d)` | — | `h x d` matrix, all zeros (`vector::resize` value-initializes `float` to `0.0f`) |
| `Matx::random(h, d)` | — | `h x d` matrix, uniform random in `[0, 1)` (`std::mt19937` + `std::uniform_real_distribution`) |
| `matx.add(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A + B` |
| `matx.sub(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A - B` |
| `matx.scale(A, s)` | — | Elementwise `A * s` |
| `matx.mul(A, B)` | `A.dims == B.num_rows()` | Matrix multiplication, `M x K` times `K x P` → `M x P` |
| `A.transpose()` | — | `M x N` → `N x M` |

Shape mismatches throw `std::invalid_argument` rather than silently producing incorrect results.

See `tests/` and `examples/` for the test suite and example applications, respectively.
