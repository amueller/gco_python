cmake_minimum_required (VERSION 2.6)
project (gco-python)

# CPACK options for debian package
set(CPACK_PACKAGE_NAME python-gco)
set(CPACK_PACKAGE_VERSION 3.0)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
set(CPACK_PACKAGE_CONTACT "amueller@ais.uni-bonn.de")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Graph cut optimization by Olga Veksler and Andrew Delong")
INCLUDE(CPack)

FIND_PACKAGE( PythonLibs )

set(MAKE_BUILD_TYPE Release)

add_library(gco SHARED
    graph.cpp
    maxflow.cpp
    GCoptimization.cpp
    LinkedBlockList.cpp
    )


execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(FILES gco.py  DESTINATION ${PYTHON_SITE_PACKAGES})
install(TARGETS gco DESTINATION lib)
