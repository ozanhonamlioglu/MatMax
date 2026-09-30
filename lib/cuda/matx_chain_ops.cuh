#pragma once

#include "../chain.hpp"
#include <vector>

float* matx_chain_add(const matx::ChainMatrix& A, const matx::ChainMatrix& B);
float* matx_chain_sub(const matx::ChainMatrix& A, const matx::ChainMatrix& B);
float* matx_chain_scale(const matx::ChainMatrix& A, float scalar);
float* matx_chain_mul(const matx::ChainMatrix& A, const matx::ChainMatrix& B);
float* matx_chain_transpose(const matx::ChainMatrix& A);

// Copies the last buffer in the collector into Host, then frees every buffer.
void matx_chain_complete(const std::vector<matx::DeviceMemCollector>& collector, float *Host);
// Frees every buffer in the collector.
void matx_chain_free(const std::vector<matx::DeviceMemCollector>& collector);
