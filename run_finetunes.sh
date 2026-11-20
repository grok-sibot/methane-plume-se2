#!/usr/bin/env bash
# Fine-tunes from the shared 'base' checkpoint (soup ingredients), two at a time.
cd /workspace/methane/src; . ../venv/bin/activate
C="--init ../runs/base/final.pt --epochs 8 --lr 3e-4 --threads 4"
env -u HF_TOKEN python train.py --name ft_s0 --seed 0 --aug 1 $C > ../runs/ft_s0.log 2>&1 &
env -u HF_TOKEN python train.py --name ft_s1 --seed 1 --aug 1 $C > ../runs/ft_s1.log 2>&1 &
wait
env -u HF_TOKEN python train.py --name ft_aug2 --seed 2 --aug 2 $C > ../runs/ft_aug2.log 2>&1 &
env -u HF_TOKEN python train.py --name ft_neg --seed 3 --aug 1 --neg_p 0.3 $C > ../runs/ft_neg.log 2>&1 &
wait
echo ALLDONE
