from distutils.core import setup
from distutils.extension import Extension
import os
import subprocess

VERSION = '0.0.1'

try:
    # Attempt to set up the cython module
    from Cython.Distutils import build_ext
    import numpy

    # Make sure the gco_src directory is up to date. Technically this should
    # occur during a build command, not during configuration
    subprocess.check_call(['make', 'gco_src'])

    gco_directory = "gco_src"

    files = ['GCoptimization.cpp', 'graph.cpp', 'LinkedBlockList.cpp',
             'maxflow.cpp']

    files = [os.path.join(gco_directory, f) for f in files]
    files.insert(0, "gco_python.pyx")

    setup(
        name='pygco',
        version=VERSION,
        install_requires=['cython', 'numpy'],
        cmdclass={'build_ext': build_ext},
        ext_modules=[Extension(
            "pygco", files, language="c++",
            include_dirs=[gco_directory, numpy.get_include()],
            library_dirs=[gco_directory],
            extra_compile_args=["-fpermissive"]
        )],
    )
except ImportError:
    # If cython or numpy are not available, then do not try to configure the
    # cython extension, just record that we need them as dependencies
    setup(
        name='pygco',
        version=VERSION,
        install_requires=['cython', 'numpy']
    )
