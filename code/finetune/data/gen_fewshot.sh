DATASET=$1
N_FEWSHOT=$2
SEED=2023
python -u gen_fewshot.py --seed=$SEED --n_sample=$N_FEWSHOT --dataset=$DATASET