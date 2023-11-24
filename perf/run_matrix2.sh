#!/usr/bin/env bash

for x in 100000 1000000 10000000; do
  for y in 16; do
    ./lorenz_perf.job -t 0.1 -c 0 -r ${y} -k ${x} -m performance_${x}_${y}
 done
done 
