# ZFP goodness experiments
for c in -1 0 1 2 4 8 16 32 64 128 512
do
    for r in 8 12 16 24 32 48 64
    do
        ./lorenz96.job -w 10 -t 6 -r $r -e 11 -a zfp -c $c --dt 0.003
        sleep 15
    done
done

# BitRound performance experiments
for c in -1 0 1 2 4 8 16 32 64 128 512
do
    for b in 8 12 16 24 32 48 52
    do
        ./lorenz96.job -w 10 -t 6 -b $b -e 11 -a bitround -c $c --dt 0.003
        sleep 15
    done
done

# ZFP GPU benchmark experiments
for k in 100 1000 10000 100000 1000000 10000000 20000000 40000000
do
    for r in 8 12 16 24 32 48 64
    do
        ./lorenz96.job -w 10 -t 1 -r $r -e 1 -a zfp -c 1 -k $k --dt 0.01
        sleep 15
    done
done

# ZFP CPU benchmark experiments
for k in 100000 1000000 10000000
do
    for r in 16
    do
        ./lorenz96.job -w 10 -t 1 -r $r -e 1 -a zfp -c 0 -k $k --dt 0.01
        sleep 15
    done
done

# BitRound GPU benchmark experiments
for k in 100 1000 10000 100000 1000000 10000000 20000000 40000000
do
    for b in 8 12 16 24 32 48 52
    do
        ./lorenz96.job -w 10 -t 1 -b $b -e 1 -a bitround -c 1 -k $k --dt 0.01
        sleep 15
    done
done

# BitRound CPU benchmark experiments
for k in 100000 1000000 10000000
do
    for b in 16
    do
        ./lorenz96.job -w 10 -t 1 -b $b -e 1 -a zfp -c 0 -k $k --dt 0.01
        sleep 15
    done
done
