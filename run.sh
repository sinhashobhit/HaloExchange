#!/bin/sh
~/UGP/allocator/src/allocator.out 64 8
make #Compiling our src.c file
rm -f data.txt #Deleting the data.txt in case it exists, which happens on reruns of the script.
for execution in 1 2 3 4 5
do
        for P in 16 36 49 64
        do
                for N in 16 32 64 128 256 512 1024
                do
                        echo "Running configuration ${execution}-P=${P}, N=${N}^2"
                        mpiexec -np $P -hostfile hosts ./halo $N 50 >> data.txt
                done
        done
done
echo "Simulation Results have been saved in data.txt"
echo "Generating Plots, The Script requires matplotlib, pandas, runtime, seaborn packages."
python3 plot.py data.txt
echo "Plots files have been generated"