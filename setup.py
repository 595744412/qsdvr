from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qsdvr",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "qsdvr",
            ["kernel/qsdvr.cpp"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
