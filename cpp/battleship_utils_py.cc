#include "pybind11/include/pybind11/eigen.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

#include "drake/common/autodiff.h"
PYBIND11_NUMPY_OBJECT_DTYPE(drake::AutoDiffXd)

#include "battleship_utils.h"

namespace py = pybind11;

using drake::AutoDiffXd;

namespace battleship_utils {

PYBIND11_MODULE(battleship_utils_py, m) {
  m.doc() = "...";

  py::class_<Ship<double>>(m, "Ship")
      .def(py::init<int, double, double, double, std::vector<double>>())
      .def("GetTfMat", &Ship<double>::GetTfMat)
      .def("GetPoints", &Ship<double>::GetPoints, py::arg("spacing") = 0.5,
           py::arg("side_length") = 1.0)
      .def("GetPointsInWorldFrame", &Ship<double>::GetPointsInWorldFrame,
           py::arg("spacing") = 0.5, py::arg("side_length") = 1.0)
      .def("GetSignedDistanceToPoints",
           &Ship<double>::GetSignedDistanceToPoints)
      .def("get_color", &Ship<double>::get_color)
      .def("get_x", &Ship<double>::get_x)
      .def("set_x", &Ship<double>::set_x)
      .def("get_y", &Ship<double>::get_y)
      .def("set_y", &Ship<double>::set_y)
      .def("get_theta", &Ship<double>::get_theta)
      .def("set_theta", &Ship<double>::set_theta)
      .def("get_theta", &Ship<double>::get_theta)
      .def("get_length", &Ship<double>::get_length);

  py::class_<Ship<AutoDiffXd>>(m, "ShipAutodiff")
      .def(py::init<int, AutoDiffXd, AutoDiffXd, AutoDiffXd,
                    std::vector<double>>())
      .def("GetTfMat", &Ship<AutoDiffXd>::GetTfMat)
      .def("GetPoints", &Ship<AutoDiffXd>::GetPoints, py::arg("spacing") = 0.5,
           py::arg("side_length") = 1.0)
      .def("GetPointsInWorldFrame", &Ship<AutoDiffXd>::GetPointsInWorldFrame,
           py::arg("spacing") = 0.5, py::arg("side_length") = 1.0)
      .def("GetSignedDistanceToPoints",
           &Ship<AutoDiffXd>::GetSignedDistanceToPoints)
      .def("get_color", &Ship<AutoDiffXd>::get_color)
      .def("get_x", &Ship<AutoDiffXd>::get_x)
      .def("set_x", &Ship<AutoDiffXd>::set_x)
      .def("get_y", &Ship<AutoDiffXd>::get_y)
      .def("set_y", &Ship<AutoDiffXd>::set_y)
      .def("get_theta", &Ship<AutoDiffXd>::get_theta)
      .def("set_theta", &Ship<AutoDiffXd>::set_theta)
      .def("get_length", &Ship<AutoDiffXd>::get_length);

  py::class_<Board>(m, "Board").def(py::init<>());

  m.attr("__version__") = "dev";
}
}