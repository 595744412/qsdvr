#include <render.h>
#include <camera.h>
#include <surface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.doc() = "render kernel";
      m.def("GridInterpolationForward",
            &GridInterpolationForward);
      m.def("GridInterpolationBackward",
            &GridInterpolationBackward);
      m.def("LayerToGridForward",
            &LayerToGridForward);
      m.def("LayerToGridBackward",
            &LayerToGridBackward);
      m.def("SampleSDFForward",
            &SampleSDFForward);
      m.def("SampleSDFBackward",
            &SampleSDFBackward);
      m.def("SampleNormalForward",
            &SampleNormalForward);
      m.def("SampleNormalBackward",
            &SampleNormalBackward);
      m.def("ShaderForward",
            &ShaderForward);
      m.def("ShaderBackward",
            &ShaderBackward);
      m.def("RayAggregateForward",
            &RayAggregateForward);
      m.def("RayAggregateBackward",
            &RayAggregateBackward);
      m.def("RayCut",
            &RayCut);
      m.def("GenerateRayPoints",
            &GenerateRayPoints);
      py::class_<PyCameraArgs>(m, "CameraArgs")
          .def(py::init<>())
          .def_readwrite("width", &PyCameraArgs::width)
          .def_readwrite("height", &PyCameraArgs::height)
          .def_readwrite("rotation", &PyCameraArgs::rotation)
          .def_readwrite("translation", &PyCameraArgs::translation)
          .def_readwrite("fx", &PyCameraArgs::fx)
          .def_readwrite("fy", &PyCameraArgs::fy)
          .def_readwrite("cx", &PyCameraArgs::cx)
          .def_readwrite("cy", &PyCameraArgs::cy)
          .def_readwrite("near", &PyCameraArgs::near)
          .def_readwrite("far", &PyCameraArgs::far)
          .def_readwrite("pixelCount", &PyCameraArgs::pixelCount)
          .def_readwrite("sdfCount", &PyCameraArgs::sdfCount)
          .def_readwrite("renderCount", &PyCameraArgs::renderCount);
}