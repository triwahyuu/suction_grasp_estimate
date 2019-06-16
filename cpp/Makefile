all: build

trace:
	@python model_trace.py

build: example.cpp CMakeLists.txt
	@cd build/ && cmake .. && make -j4