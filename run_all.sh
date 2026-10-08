#!/usr/bin/env bash
# Full pipeline, small default scale (~30-40 min on a laptop). Edit numbers to scale up.
set -e
python doctor.py
python fetch_dataset.py --source synthetic --n 400        # or: --source compile --n 600
python validate_passes.py
python generate_labels.py --max-modules 350 --seqs-per-module 60
python train.py --epochs 30 --models 3
python grammar_ablation.py --modules 12 --seqs 12
python evaluate.py --n-test 20 --ablate
python plots.py
