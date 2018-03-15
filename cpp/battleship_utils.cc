#include "battleship_utils.h"

#include <drake/common/autodiff.h>

#include <unistd.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace std;
using namespace Eigen;
using namespace battleship_utils;

// Project point p onto bounded line segment
// v w
// Ref
// https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
template <typename Scalar>
Matrix<Scalar, 2, 1> project_onto_line_segment(Matrix<Scalar, 2, 1> v,
                                               Matrix<Scalar, 2, 1> w,
                                               Matrix<Scalar, 2, 1> p) {
  Scalar l2 = (w - v).transpose() * (w - v);

  auto t = max(0.0, min(1.0, ((p - v).transpose() * (w - v))[0] / l2));
  return v + t * (w - v);
}

template <typename Scalar>
Scalar get_signed_distance_to_line_segment(Matrix<Scalar, 2, 1> v,
                                           Matrix<Scalar, 2, 1> w,
                                           Matrix<Scalar, 2, 1> p) {
  auto projected_point = project_onto_line_segment(v, w, p);
  auto error = projected_point - p;

  // Get sign by dotting onto a vector orthogonal to the line,
  // assuming it's directed v->w
  Scalar l2 = (w - v).transpose() * (w - v);
  Matrix<Scalar, 2, 1> cross_dir;
  cross_dir(0) = (w - v)(1) / l2;
  cross_dir(1) = -(w - v)(0) / l2;
  if ((error.transpose() * cross_dir)[0] >= 0.)
    return (error.transpose() * error)[0];
  else
    return -1.0 * (error.transpose() * error)[0];
}

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
    int n_pts = 4 + 2 * ceil((side_length + length_ - 1) / spacing - 1) +
                2 * ceil(side_length / spacing - 1);
    Matrix<Scalar, 2, Dynamic> points(2, n_pts);

    int k = 0;
    for (int i = 0; i < 4; i++) {
      auto c1 = corners.col(i);
      auto c2 = corners.col((i + 1) % 4);

      points.col(k) = c1;
      k++;

      Scalar edge_length = sqrt((c2 - c1).transpose() * (c2 - c1));
      for (Scalar interp = spacing / edge_length; interp < 0.999;
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
  Matrix<Scalar, 2, Dynamic> intermed =
      tf.block(0, 0, 2, 2) * GetPoints(spacing, side_length);
  // Colwise isn't happy with me...
  for (int i = 0; i < intermed.cols(); i++)
    intermed.col(i) += tf.block(0, 2, 2, 1);
  return intermed;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, 1> Ship<Scalar>::GetSignedDistanceToPoints(
    const Matrix<Scalar, 2, Dynamic> points) {
  Matrix<Scalar, 4, Dynamic> phi_all(4, points.cols());
  Matrix<Scalar, Dynamic, 1> phi_out(points.cols(), 1);

  auto corners = GetPointsInWorldFrame(length_, 1.0);

  for (int i = 0; i < 4; i++) {
    auto c1 = corners.col(i);
    auto c2 = corners.col((i + 1) % 4);
    for (int k = 0; k < points.cols(); k++) {
      phi_all(i, k) =
          get_signed_distance_to_line_segment<Scalar>(c1, c2, points.col(k));
    }
  }

  // Take min abs distance with sig
  for (int k = 0; k < points.cols(); k++) {
    phi_out(k) = phi_all.col(k).cwiseAbs().minCoeff();
    if (phi_all.col(k).maxCoeff() < 0.) phi_out(k) *= -1.;
  }

  return phi_out;
}

template class Ship<double>;
template class Ship<drake::AutoDiffXd>;

Board::Board() { printf("ASDFASDFASDF\n"); }