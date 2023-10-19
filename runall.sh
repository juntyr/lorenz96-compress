for c in -1 0 1 2 4 8 16 32 64 128 512
do
    for r in 8 12 16 24 32 48 64
    do
        ./lorenz96.job -w 10 -t 10 -r $r -e 11 -c $c
    done
done
