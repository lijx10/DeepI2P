import setuptools
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension


setup(name='ball_query',
      ext_modules=[CUDAExtension('ball_query', ['ball_query.cpp', 'ball_query_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})
