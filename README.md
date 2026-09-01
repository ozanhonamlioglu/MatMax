# MatMax

A small CUDA-accelerated matrix library written from scratch in C++20/CUDA, with a hand-rolled logistic regression built entirely on top of it — no cuBLAS, no Eigen, no ML frameworks. Every matrix operation the training loop needs (multiply, transpose, elementwise add/sub, scalar scale) is a real CUDA kernel.

## Why this project exists

This is an exploration that how far you can get building GPU-accelerated linear algebra primitives from the ground up, then using nothing but those primitives to implement a real (if simple) machine learning algorithm — forward pass, loss, gradient, and weight update, all expressed as matrix ops dispatched to the GPU.

## Project structure

```
matmax/
├── CMakeLists.txt
├── main.cpp                    # entry point for the test suite binary
├── test.cpp                    # unit tests for the matrix library
├── lib/
│   ├── matx.hpp                 # Matrix struct + Matx op-dispatch class
│   ├── matx.cpp                 # Matx method implementations (host side)
│   └── cuda/
│       ├── matx_ops.cuh         # CUDA kernel declarations
│       └── matx_ops.cu          # CUDA kernels + host launch wrappers
└── ml/
    └── logistic-regression.cpp  # from-scratch logistic regression demo
```

Two executables come out of the build: `matmax` (the library + its test suite) and `logreg` (the logistic regression demo).

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

./matmax   # runs the matrix library test suite
./logreg   # trains & validates logistic regression
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so it targets whatever GPU is in the machine doing the build.

## The matrix library

### `Matrix` (`lib/matx.hpp`)

A minimal row-major matrix: a flat `std::vector<float>` plus `dims`, the length of one row.

| Member | Description |
|---|---|
| `num_rows()` | `mtx.size() / dims` |
| `get_cell_at(row, col)` | Bounds-checked element access |
| `transpose()` | Returns a new transposed matrix (host-side; it's pure index remapping, not worth a kernel) |

### `Matx` — the CUDA-backed operations

Every op below allocates device memory, copies operands to the GPU, launches a kernel, copies the result back, and frees device memory. Kernels use a flat 1D grid-stride-free launch (`threadIdx.x + blockIdx.x * blockDim.x`), 256 threads per block, with a bounds check so sizes that aren't clean multiples of the block size don't read/write out of range.

| Method | Shape rule | What it does |
|---|---|---|
| `Matx::zeros(h, d)` | — | `h x d` matrix, all zeros (`vector::resize` value-initializes `float` to `0.0f`) |
| `Matx::random(h, d)` | — | `h x d` matrix, uniform random in `[0, 1)` (`std::mt19937` + `std::uniform_real_distribution`) |
| `matx.add(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A + B` |
| `matx.sub(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A - B` |
| `matx.scale(A, s)` | — | Elementwise `A * s` |
| `matx.mul(A, B)` | `A.dims == B.num_rows()` | Real matrix multiplication, `M x K` times `K x P` → `M x P` |
| `A.transpose()` | — | `M x N` → `N x M` |

Shape mismatches throw `std::invalid_argument` rather than silently producing garbage.

### Test suite (`test.cpp`)

Each op gets a small, hand-computed check. `test_compare<T>` is a C++20 concept-constrained helper restricted to `int`/`float`, and uses `std::source_location` as a default argument so it automatically prints which test function called it — no logging boilerplate needed at each call site:

```cpp
void test_compare(IntOrFloat auto x, IntOrFloat auto y,
                   const std::source_location& loc = std::source_location::current());
```

Sample run:

```
void test_mat_add() -> OK
void test_mat_mul() -> OK
void test_mat_mul() -> OK
void test_mat_mul_transpose() -> OK
void test_mat_mul_transpose() -> OK
void test_mat_sub() -> OK
void test_mat_scale() -> OK
void zeros_test() -> OK
void random_test() -> OK
void random_test() -> OK
void random_test() -> OK
```

## Logistic regression from scratch (`ml/logistic-regression.cpp`)

A binary classifier trained with full-batch gradient descent, where every matrix operation goes through `Matx`.

**Dataset.** Synthetic and linearly separable: each sample's features are drawn from a Gaussian centered at `(-2, -2, ...)` for class 0 or `(2, 2, ...)` for class 1, alternating labels, seeded for reproducibility. A constant bias column of `1.0` is appended so the bias term is learned as just another weight.

**Model.** `p = sigmoid(X · w)`, where `w` is initialized to small random values around 0 via `Matx::random`.

**Training loop**, entirely matrix ops:

```cpp
Matrix Xt = ds.X.transpose();               // (D+1) x N, computed once

for (int epoch = 0; epoch < epochs; ++epoch) {
  Matrix z = matx.mul(ds.X, w);             // forward: N x 1
  Matrix p = apply_sigmoid(z);              // activation (elementwise, hand-rolled)

  Matrix error = matx.sub(p, ds.y);         // dL/dz, N x 1
  Matrix grad  = matx.mul(Xt, error);       // Xᵀ · error, (D+1) x 1
  Matrix delta = matx.scale(grad, -lr / n_samples);

  w = matx.add(w, delta);                   // gradient descent step
}
```

Sigmoid and binary cross-entropy are the only pieces done as plain loops — they're elementwise nonlinear/log-based math, not something a `Matx` primitive covers.

**Validation.** After training, the forward pass is re-run on the same dataset to sanity-check convergence (this is a from-scratch demo, not a benchmark — there's no held-out test split).

### Sample output

```
epoch 0 -> loss: 0.646289
epoch 1000 -> loss: 0.00214259
epoch 2000 -> loss: 0.00107136
epoch 3000 -> loss: 0.000714659
epoch 4000 -> loss: 0.000536315
epoch 4999 -> loss: 0.000429373
training took 5850 ms for 5000 epochs
final loss: 0.000429287
final accuracy: 100%
learned weights: 0.0352103 0.0365744 0.0343283 0.0318164 0.0364427 ...
```

(3000 samples, 120 features, learning rate `0.001`, 5000 epochs — GTX 1050 Ti.)

That ~5.8s for 5000 epochs is mostly `cudaMalloc`/`cudaMemcpy` overhead — each epoch launches 3 separate kernels, each with its own host↔device round trip. This library optimizes for clarity over performance (device memory isn't persisted across calls), which is a reasonable next thing to fix if speed becomes the point.
