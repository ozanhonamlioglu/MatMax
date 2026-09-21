#pragma once

void matx_add(float* A, float* B, float* Buffer, int N);

void matx_sub(float* A, float* B, float* Buffer, int N);

void matx_scale(float* A, float* Buffer, float scalar, int N);

void matx_mul(float* A, float* B, float* Buffer, int M, int K, int P);

// bool matx_equal(float* A, float* B, int length);

// Matrix Generation
void matx_randomf(int length, float* out);

// void matx_zeros();