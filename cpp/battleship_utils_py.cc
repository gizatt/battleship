#include "pybind11/include/pybind11/eigen.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

#include "battleship_utils.h"

namespace py = pybind11;

namespace battleship_utils {

PYBIND11_MODULE(battleship_utils_py, m) {
  py::module::import("pydrake.rbtree");

  m.doc() = "...";

  py::class_<Ship<double>>(m, "Ship")
      .def(py::init<int, double, double, double, std::vector<double>>())
      .def("GetTfMat", &Ship<double>::GetTfMat)
      .def("GetPoints", &Ship<double>::GetPoints, py::arg("spacing") = 0.5,
           py::arg("side_length") = 1.0)
      .def("GetPointsInWorldFrame", &Ship<double>::GetPointsInWorldFrame,
           py::arg("spacing") = 0.5, py::arg("side_length") = 1.0)
      .def("GetSignedDistanceToPoint", &Ship<double>::GetSignedDistanceToPoint)
      .def("get_color", &Ship<double>::get_color);

  py::class_<Board>(m, "Board").def(py::init<>());

  m.attr("__version__") = "dev";
}
}