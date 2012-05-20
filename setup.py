from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

gco_directory = "../gco/"

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pygco", ["gco_python.pyx"], language="c++",
        include_dirs=[gco_directory], library_dirs=[gco_directory],
        libraries=['gco'])])
