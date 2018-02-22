#include "pybind11/include/pybind11/eigen.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

#include "battleship_utils.h"

namespace py = pybind11;

namespace battleship_utils {

PYBIND11_MODULE(battleship_utils_py, m) {
  py::module::import("pydrake.rbtree");

  m.doc() = "...";

  py::class_<Ship>(m, "Ship")
      .def(py::init<>());

  py::class_<Board>(m, "Board")
      .def(py::init<>());
      

  m.attr("__version__") = "dev";
}

}