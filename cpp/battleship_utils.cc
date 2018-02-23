#include "battleship_utils.h"

#include <unistd.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace std;
using namespace Eigen;
using namespace battleship_utils;

template <typename Scalar>
Ship<Scalar>::Ship(int length, Scalar x, Scalar y, Scalar theta,
                   std::vector<double> color)
    : length_(length), x_(x), y_(y), theta_(theta), color_(color) {}

template <typename Scalar>
Matrix<Scalar, 3, 3> Ship<Scalar>::GetTfMat() const {
  Matrix<Scalar, 3, 3> tf;
  tf << cos(theta_), -sin(theta_), x_, sin(theta_), cos(theta_), y_, 0., 0., 1.;
  return tf;
}

template <typename Scalar>
Matrix<Scalar, 2, Dynamic> Ship<Scalar>::GetPoints(double spacing,
                                                   double side_length) {
  auto this_sig = GetPointCallSignature(spacing, side_length);

  const auto it = cached_points_.find(this_sig);
  if (it == cached_points_.end()) {
    // Compute points.

    Matrix<Scalar, 2, 4> corners;
    corners << -side_length / 2, -side_length / 2.,
        side_length / 2. + (length_ - 1), side_length / 2. + (length_ - 1),
        -side_length / 2, side_length / 2., side_length / 2., -side_length / 2.;

    // Will return 4 corners, plus additional points from interpolating
    // the edges.
    int n_pts =
        4 + 2 * ceil(side_length / spacing - 1) + 2 * ceil(1. / spacing - 1);
    Matrix<Scalar, 2, Dynamic> points(2, n_pts);

    int k = 0;
    for (int i = 0; i < 4; i++) {
      auto c1 = corners.col(i);
      auto c2 = corners.col(i);

      points.col(k) = c1;
      k++;

      double edge_length = sqrt((c2 - c1).norm());
      for (double interp = spacing / edge_length; interp < 1.;
           interp += spacing / edge_length) {
        points.col(k) = c1 * (1. - interp) + c2 * interp;
        k++;
      }
    }

    cached_points_[this_sig] = points;
    return points;
  } else {
    return it->second;
  }
}

template <typename Scalar>
Matrix<Scalar, 2, Dynamic> Ship<Scalar>::GetPointsInWorldFrame(
    double spacing, double side_length) {
  Matrix<Scalar, 3, 3> tf = GetTfMat();
  Matrix<Scalar, 2, Dynamic> intermed = tf.block(0, 0, 2, 2) * GetPoints(spacing, side_length);
  // Colwise isn't happy with me...
  for (int i = 0; i < intermed.cols(); i++)
  	intermed.col(i) += tf.block(0, 2, 2, 1);
  return intermed;
}

template <typename Scalar>
void Ship<Scalar>::GetSignedDistanceToPoint(
    const Eigen::Matrix<Scalar, 3, 1> point,
    Eigen::Ref<Eigen::Matrix<Scalar, 4, 1>>& phi,
    Eigen::Ref<Eigen::Matrix<Scalar, 4, 3>>& dphi_dq) {
  printf("todo");
}

template class Ship<double>;

Board::Board() { printf("ASDFASDFASDF\n"); }