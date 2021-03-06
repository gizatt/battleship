#pragma once

#include <map>
#include "drake/common/eigen_types.h"

namespace battleship_utils {

template <typename Scalar>
class Ship {
 public:
  Ship(int length, Scalar x, Scalar y, Scalar theta,
       std::vector<double> color = {1.0, 0.0, 0.0});
  ~Ship(){};

  Eigen::Matrix<Scalar, 3, 3> GetTfMat() const;

  Eigen::Matrix<Scalar, 2, Eigen::Dynamic> GetPoints(double spacing = 0.5,
                                                     double side_length = 1.0);

  Eigen::Matrix<Scalar, 2, Eigen::Dynamic> GetPointsInWorldFrame(
      double spacing = 0.5, double side_length = 1.0);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> GetSignedDistanceToPoints(
      const Eigen::Matrix<Scalar, 2, Eigen::Dynamic> point);

  Scalar get_x() { return x_; }
  Scalar get_y() { return y_; }
  Scalar get_theta() { return theta_; }

  void set_x(Scalar x) { x_ = x; }
  void set_y(Scalar y) { y_ = y; }
  void set_theta(Scalar theta) { theta_ = theta; }

  int get_length() { return length_; }
  std::vector<double> get_color() { return color_; }

 private:
  Scalar x_;
  Scalar y_;
  Scalar theta_;
  int length_;
  std::vector<double> color_;

  typedef std::pair<double, double> GetPointCallSignature;

  std::map<GetPointCallSignature, Eigen::Matrix<Scalar, 2, Eigen::Dynamic>>
      cached_points_;
};

class Board {
 public:
  Board();

 private:
};
}