from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pygco", ["gco_python.pyx"],
        language="c++", include_dirs=["../gco/"], library_dirs=["../gco/"], libraries=['gco'])])
