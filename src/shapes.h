#pragma once

#include <stdexcept>
#undef eigen_assert
#define eigen_assert(X)                     \
  do {                                      \
    if (!(X)) throw std::runtime_error(#X); \
  } while (false);

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace alphazero {

template <typename T>
using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T, size_t S>
using SizedVector = Eigen::Matrix<T, 1, S, Eigen::RowMajor>;

template <typename T>
using Matrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T, size_t R, size_t C>
using SizedMatrix = Eigen::Matrix<T, R, C, Eigen::RowMajor>;

template <typename T, int DIMS>
using Tensor = Eigen::Tensor<T, DIMS, Eigen::RowMajor>;

template <typename T, typename DIMS>
using SizedTensor = Eigen::TensorFixedSize<T, DIMS, Eigen::RowMajor>;

}  // namespace alphazero
