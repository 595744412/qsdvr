#include <torch/extension.h>

int add(int a, int b)
{
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "render kernel";
    m.def("add",
          &add,
          "add",
          py::arg("a"),
          py::arg("b"));
}