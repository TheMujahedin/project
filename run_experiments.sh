#!/bin/bash

HP="./hpatches"
MIR="../mirflickr"
SPLIT="./split.txt"
VOCAB=256
S=500
SET="validation"
SEED=42

# Τα μεγέθη της εκφώνησης
DISTRACTORS="1000 5000 10000 25000"

echo "=== Εκκίνηση Πλήρων Πειραμάτων Image Retrieval ==="

# 1. Exact Search
for D in $DISTRACTORS; do
    echo "--> [Exact] D=$D"
    ./search -hp $HP -mir $MIR -split $SPLIT -vocab $VOCAB -S $S -D $D -set $SET -seed $SEED -exact -o res_exact_D${D}.txt
done

# 2. LSH (διορθωμένο w=1.0, k=3)
for D in $DISTRACTORS; do
    echo "--> [LSH] D=$D"
    ./search -hp $HP -mir $MIR -split $SPLIT -vocab $VOCAB -S $S -D $D -set $SET -seed $SEED -lsh -k 3 -L 8 -w 1.0 -o res_lsh_D${D}.txt
done

# 3. Hypercube (διορθωμένο w=1.0)
for D in $DISTRACTORS; do
    echo "--> [Hypercube] D=$D"
    ./search -hp $HP -mir $MIR -split $SPLIT -vocab $VOCAB -S $S -D $D -set $SET -seed $SEED -hypercube -kproj 10 -M 2000 -probes 50 -w 1.0 -o res_cube_D${D}.txt
done

# 4. IVFFlat
for D in $DISTRACTORS; do
    echo "--> [IVFFlat] D=$D"
    ./search -hp $HP -mir $MIR -split $SPLIT -vocab $VOCAB -S $S -D $D -set $SET -seed $SEED -ivfflat -nprobe 10 -o res_ivfflat_D${D}.txt
done

# 5. IVFPQ
for D in $DISTRACTORS; do
    echo "--> [IVFPQ] D=$D"
    ./search -hp $HP -mir $MIR -split $SPLIT -vocab $VOCAB -S $S -D $D -set $SET -seed $SEED -ivfpq -nprobe 10 -M 16 -nbits 8 -o res_ivfpq_D${D}.txt
done

echo "Όλα τα πειράματα ολοκληρώθηκαν επιτυχώς!"
