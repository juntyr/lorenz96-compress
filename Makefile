.PHONY: clean purge

all: lorenz96

zfp/Makefile:
	git submodule init
	git submodule update

zfp/build/lib64/libzfp.so: zfp/Makefile
	mkdir -p zfp/build
	cd zfp/build && cmake -DZFP_WITH_HIP=ON -DZFP_WITH_OPENMP=OFF -DHIP_PATH=/opt/rocm/hip -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DBUILD_TESTING=OFF ../
	cd zfp/build && make
	cd zfp/build/src && CC -DZFP_ROUNDING_MODE=ZFP_ROUND_NEVER -DZFP_WITH_HIP -Dzfp_EXPORTS -I../../include -O3 -DNDEBUG --offload-arch=gfx90a -fPIC -std=gnu++14 -o CMakeFiles/zfp.dir/hip/interface.cpp.o -x hip -c ../../src/hip/interface.cpp
	cd zfp/build && make

lorenz96: lorenz96.cpp compress.hpp decompress.hpp zfp/build/lib64/libzfp.so
	CC -xhip -o lorenz96 lorenz96.cpp --offload-arch=gfx90a -Izfp/include -Lzfp/build/lib64 -lzfp -Wl,-rpath,zfp/build/lib64

clean:
	rm -f *.o
	rm -f lorenz96

purge: clean
	rm -f *.out
	rm -f state_*
	git submodule deinit --all
