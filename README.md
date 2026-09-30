# MatMax

MatMax is a dependency-free, CUDA-accelerated matrix library for C++. Core operations (matrix multiplication, transpose, elementwise addition/subtraction, scalar scaling) are each implemented as CUDA kernels.

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
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so it targets whatever GPU is in the machine doing the build.

## Using this library in another project

MatMax is distributed as source and consumed through CMake's `FetchContent`, so there is nothing to download or install manually — a consuming project's own build compiles `matmax` against its own compiler, CUDA Toolkit, and GPU:

```cmake
include(FetchContent)

FetchContent_Declare(
  matmax
  GIT_REPOSITORY https://github.com/ozanhonamlioglu/MatMax.git
  GIT_TAG main # pin to a released tag once one exists, e.g. v0.1.0
)
FetchContent_MakeAvailable(matmax)

target_link_libraries(your_target PRIVATE matmax)
```

Linking against `matmax` is sufficient — its include directory and its dependency on `CUDA::cudart` propagate automatically, so no additional `target_include_directories` or CUDA linkage is needed in the consuming project. Include the library with:

```cpp
#include "lib/matx.hpp"
#include "lib/chain.hpp" // only for chained operations
```

Everything lives under the `matx` namespace.

## The matrix library

### `matx::Matrix` (`lib/matx.hpp`)

A row-major matrix: a flat `std::vector<float>` plus `dims`, the length of one row.

| Member | Description |
|---|---|
| `num_rows()` | `mtx.size() / dims` |
| `get_cell_at(row, col)` | Bounds-checked element access |
| `transpose()` | Returns a new transposed matrix (host-side index remapping) |

### `matx::Ops` — the CUDA-backed operations

Every operation below allocates device memory, copies operands to the GPU, launches a kernel, copies the result back, and frees device memory. Kernels use a flat 1D launch (`threadIdx.x + blockIdx.x * blockDim.x`), 256 threads per block, with a bounds check so sizes that aren't clean multiples of the block size don't read/write out of range.

| Method | Shape rule | What it does |
|---|---|---|
| `Ops::zeros(h, d)` | — | `h x d` matrix, all zeros (`vector::resize` value-initializes `float` to `0.0f`) |
| `Ops::randomf(h, d)` | — | `h x d` matrix, uniform random in `[0, 1)` (`curand`, seeded from the clock by default) |
| `Ops::add(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A + B` |
| `Ops::sub(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A - B` |
| `Ops::scale(A, s)` | — | Elementwise `A * s` |
| `Ops::mul(A, B)` | `A.dims == B.num_rows()` | Matrix multiplication, `M x K` times `K x P` → `M x P` |
| `Ops::is_equal(A, B)` | `A.dims == B.dims`, same total size | `true` iff every element matches |
| `Ops::transpose(A)` | — | `M x N` → `N x M`, on the device |
| `A.transpose()` | — | `M x N` → `N x M`, on the host |

Shape mismatches throw `std::invalid_argument` rather than silently producing incorrect results.

See `tests/` for the test suite.

## Chained operations

### `matx::ChainOps` (`lib/chain.hpp`)

The one-shot `Ops` pay a host↔device round trip on every call. `ChainOps` keeps results on the device instead, so a sequence of operations only copies back to the host once, at the end.

Operands are `matx::ChainMatrix`: either a host pointer (`mtx`) or a device pointer (`device`), plus `dims` and `rows`. Host operands are uploaded by the op that consumes them; results of earlier ops are already on the device and are reused as is.

```cpp
matx::ChainMatrix cA { .mtx = A.mtx.data(), .device = nullptr, .dims = A.dims, .rows = A.num_rows() };
matx::ChainMatrix cB { .mtx = B.mtx.data(), .device = nullptr, .dims = B.dims, .rows = B.num_rows() };

matx::ChainOps chain;
matx::ChainMatrix sum = chain.add(cA, cB);
chain.scale(sum, 2.5f);
matx::Matrix result = chain.complete(); // (A + B) * 2.5
```

| Method | Shape rule | What it does |
|---|---|---|
| `add(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A + B` |
| `sub(A, B)` | `A.dims == B.dims`, same total size | Elementwise `A - B` |
| `scale(A, s)` | — | Elementwise `A * s` |
| `mul(A, B)` | `A.dims == B.rows` | Matrix multiplication, `M x K` times `K x P` → `M x P` |
| `transpose(A)` | — | `M x N` → `N x M` |
| `complete()` | — | Copies the **last** op's result to the host as a `matx::Matrix` and frees all device memory of the chain |

Every result stays on the device until `complete()` is called, and every `ChainMatrix` returned by the chain is invalid after it. If `complete()` is never called, the destructor frees the device memory. `ChainOps` is not copyable.
