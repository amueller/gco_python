gco_python: gco_src
	python setup.py build_ext -i

gco-v3.0.zip:
	wget http://mouse.cs.uwaterloo.ca/code/gco-v3.0.zip

gco_src: gco-v3.0.zip
	mkdir gco_src
	cd gco_src && unzip ../gco-v3.0.zip
