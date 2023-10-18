.PHONY: clean purge

lorenz96: lorenz96.cpp
	CC -xhip -o lorenz96 lorenz96.cpp -Izfp/include -Lzfp/build/lib64 -lzfp -Wl,-rpath,zfp/build/lib64

clean:
	rm -f *.o
	rm -f lorenz96

purge: clean
	rm -f *.out
	rm -f state_*
