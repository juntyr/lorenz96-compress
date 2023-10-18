.PHONY: clean purge

lorenz96: lorenz96.cpp
	CC -xhip -o lorenz96 lorenz96.cpp

clean:
	rm -f *.o
	rm -f lorenz96

purge: clean
	rm -f *.out
	rm -f state_*_*
