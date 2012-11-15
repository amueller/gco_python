PY_INC=/usr/include/python2.7

gco_python: gco.so
	python setup.py build_ext -i

gco-v3.0.zip:
	wget http://vision.csd.uwo.ca/code/gco-v3.0.zip

gco_src: gco-v3.0.zip
	mkdir gco_src
	cd gco_src && unzip ../gco-v3.0.zip

gco.so: gco_src
	g++ -fPIC -shared -Lgco_src -Igco_src gco_src/GCoptimization.cpp gco_src/LinkedBlockList.cpp gco_src/graph.cpp gco_src/maxflow.cpp -o gco.so
