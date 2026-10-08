@echo off
REM Full pipeline, small default scale. Run inside the environment where opt/clang work.
python doctor.py || exit /b 1
python fetch_dataset.py --source synthetic --n 400 || exit /b 1
python validate_passes.py || exit /b 1
python generate_labels.py --max-modules 350 --seqs-per-module 60 || exit /b 1
python train.py --epochs 30 --models 3 || exit /b 1
python grammar_ablation.py --modules 12 --seqs 12 || exit /b 1
python evaluate.py --n-test 20 --ablate || exit /b 1
python plots.py
