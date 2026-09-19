#!/bin/bash

seeds=(9771 2000 324)

for seed in "${seeds[@]}"
do
    echo "Running seed = $seed"

    python finetuned-bert.py --seed $seed 2>&1 | tee log_seed_${seed}.txt

    echo "Waiting for GPU to cool down..."
    sleep 10
done

echo "All seeds completed!"
