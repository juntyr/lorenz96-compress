for c in -1 0 1 2 4 8 16 32 64 128 512
do
    for r in 8 12 16 24 32 48 64
    do
        ./lorenz96.job -w 10 -t 6 -r $r -e 11 -a zfp -c $c --dt 0.003
    done
done

for c in -1 0 1 2 4 8 16 32 64 128 512
do
    for b in 8 12 16 24 32 48 52
    do
        ./lorenz96.job -w 10 -t 6 -b $b -e 11 -a bitround -c $c --dt 0.003
    done
done
