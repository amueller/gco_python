GCO_LIB=../gco/
GCO_INC=../gco/
PY_INC=/usr/include/python2.7

gco_python:
	g++ -fPIC -shared gco_python.cpp -I$(GCO_INC) -L$(GCO_LIB) -lgco -lboost_python -I$(PY_INC) -o _gco_python.so
