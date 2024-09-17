#!/bin/sh 
entry=$1
# for i in {0..9}; do
#     echo "processing $i random"
#     mpirun -n $((1 + 2 * $i)) ./build/examples/mp/sparse_benchmark ./dest/ 10000 $((10000 * (1 + 2 * $i))) 0.01
# done

# for i in {1..8}; do
#     echo "processing $i bench weak"
#     mpirun -n $i ./build/examples/mp/sparse_benchmark ./dest/ $(($i * 50000)) 4000 0
# done

for i in {1..8}; do
    echo "processing $i bench strong"
    mpirun -n $i ./build/examples/mp/sparse_benchmark ./dest/ 100000 10000 0
done


# for i in {0..9}; do
#     echo "processing $i bench weak"
#     mpirun -n $((1 + 2 * $i)) ./build/examples/mp/sparse_benchmark ./dest/ $(((1 + 2 * $i) * 100000)) 2000 0
# done

# for i in {0..9}; do
#     echo "processing $i bench strong"
#     mpirun -n $((1 + 2 * $i)) ./build/examples/mp/sparse_benchmark ./dest/ 100000 10000 0
# done

# for i in {0..9}; do
#     echo "processing $i $entry"
#     mpirun -n $((1 + 2 * $i)) ./build/examples/mp/sparse_benchmark ./dest/ $entry
# done