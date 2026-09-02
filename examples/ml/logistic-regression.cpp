#include "lib/matx.hpp"

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <ranges>

struct Dataset {
  Matrix X; // N x (D+1), last column is the bias term (always 1)
  Matrix y; // N x 1
};

// Two Gaussian blobs around (-2,...,-2) and (2,...,2) -> linearly separable classes
Dataset generate_dataset(int n_samples, int n_features) {
  Dataset ds;
  ds.X.dims = n_features + 1;
  ds.X.mtx.resize(n_samples * ds.X.dims);

  ds.y.dims = 1;
  ds.y.mtx.resize(n_samples);

  std::mt19937 gen(42);
  std::normal_distribution<float> noise(0.0f, 1.0f);

  for(int i = 0; i < n_samples; ++i) {
    int label = i % 2;
    float center = label == 0 ? -2.0f : 2.0f;

    for(int j = 0; j < n_features; ++j) {
      ds.X.mtx[i * ds.X.dims + j] = center + noise(gen);
    }
    ds.X.mtx[i * ds.X.dims + n_features] = 1.0f; // bias column

    ds.y.mtx[i] = static_cast<float>(label);
  }

  return ds;
}

float sigmoid(float z) {
  return 1.0f / (1.0f + std::exp(-z));
}

Matrix apply_sigmoid(const Matrix& z) {
  Matrix out;
  out.dims = z.dims;
  out.mtx.resize(z.mtx.size());

  for(size_t i = 0; i < z.mtx.size(); ++i) {
    out.mtx[i] = sigmoid(z.mtx[i]);
  }

  return out;
}

float binary_cross_entropy(const Matrix& y_true, const Matrix& y_pred) {
  int n = static_cast<int>(y_true.mtx.size());
  float loss = 0.0f;

  for(int i = 0; i < n; ++i) {
    float y = y_true.mtx[i];
    float p = std::clamp(y_pred.mtx[i], 1e-7f, 1.0f - 1e-7f);
    loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
  }

  return loss / n;
}

float accuracy(const Matrix& y_true, const Matrix& y_pred) {
  int n = static_cast<int>(y_true.mtx.size());
  int correct = 0;

  for(int i = 0; i < n; ++i) {
    int predicted_label = y_pred.mtx[i] >= 0.5f ? 1 : 0;
    if(predicted_label == static_cast<int>(y_true.mtx[i])) {
      ++correct;
    }
  }

  return static_cast<float>(correct) / n;
}

int main() {
  const int n_samples = 3000;
  const int n_features = 120;
  const int epochs = 5000;
  const float lr = 0.001f;

  Dataset ds = generate_dataset(n_samples, n_features);

  // Small random init around 0
  Matrix w = Matx::random(ds.X.dims, 1);
  for(float& v : w.mtx) {
    v = (v - 0.5f) * 0.01f;
  }

  Matrix Xt = ds.X.transpose(); // (D+1) x N, X never changes so transpose once

  auto start = std::chrono::steady_clock::now();

  for(int epoch = 0; epoch < epochs; ++epoch) {
    Matrix z = Matx::mul(ds.X, w);  // N x 1
    Matrix p = apply_sigmoid(z);   // N x 1

    Matrix error = Matx::sub(p, ds.y); // N x 1

    Matrix grad = Matx::mul(Xt, error); // (D+1) x 1

    Matrix delta = Matx::scale(grad, -lr / n_samples);

    w = Matx::add(w, delta);

    if(epoch % 1000 == 0 || epoch == epochs - 1) {
      float loss = binary_cross_entropy(ds.y, p);
      std::cout << "epoch " << epoch << " -> loss: " << loss << std::endl;
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "training took " << elapsed_ms << " ms for " << epochs << " epochs" << std::endl;

  // Validate: forward pass again on the same dataset the model trained on
  Matrix z_final = Matx::mul(ds.X, w);
  Matrix p_final = apply_sigmoid(z_final);

  float final_loss = binary_cross_entropy(ds.y, p_final);
  float final_acc = accuracy(ds.y, p_final);

  std::cout << "final loss: " << final_loss << std::endl;
  std::cout << "final accuracy: " << final_acc * 100.0f << "%" << std::endl;

  std::cout << "learned weights: ";
  for(float v : w.mtx | std::views::take(5)) {
    std::cout << v << " ";
  }
  std::cout << "...";
  std::cout << std::endl;

  return 0;
}
