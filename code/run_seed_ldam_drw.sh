#!/bin/bash

seeds=(9771 2000 324)
PYTHON_BIN="${PYTHON_BIN:-python3}"

for seed in "${seeds[@]}"; do
    echo "Running seed = $seed"

    "${PYTHON_BIN}" finetuned-bert-ldam-drw.py --seed $seed 2>&1 | tee log_ldam_drw_seed_${seed}.txt

    echo "Seed $seed completed."
    echo "Waiting for GPU to cool down..."
    sleep 10
done

echo "All LDAM-DRW seeds completed!"
