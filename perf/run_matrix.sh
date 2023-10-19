#!/usr/bin/env bash

for x in 100 1000 10000 100000 1000000 10000000 20000000 40000000; do
  for y in 4 8 16 32 64; do
    ./lorenz_perf.job -t 0.1 -c 1 -r ${y} -k ${x} -m performance/performance_${x}_${y}
 done
done 
