.PHONY: clean

test_zfp: main.cpp
	hipcc --offload-arch=gfx90a -I../zfp/include main.cpp -L../zfp/build/lib64 -lzfp -Wl,-rpath,../zfp/build/lib64 -o main -g

clean:
	rm -f *.o
	rm -f slurm-*
	rm -f main core
