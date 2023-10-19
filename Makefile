.PHONY: clean purge

all: lorenz96 lorenz_perf

lorenz96: lorenz96.cpp
	CC -xhip -o lorenz96 lorenz96.cpp -Izfp/include -Lzfp/build/lib64 -lzfp -Wl,-rpath,zfp/build/lib64

lorenz_perf: lorenz_perf.cpp
	CC -xhip -o lorenz_perf lorenz_perf.cpp -Izfp/include -Lzfp/build/lib64 -lzfp -Wl,-rpath,/pfs/lustrep1/projappl/project_462000376/compress/zfp/build/lib64

clean:
	rm -f *.o
	rm -f lorenz96

purge: clean
	rm -f *.out
	rm -f state_*
