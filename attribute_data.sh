#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --job-name=translate_data

module purge

module load Python/3.10.4-GCCcore-11.3.0

source $HOME/.envs/rude_nmt/bin/activate

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# move the cached datasets to the /scratch directory so that we have more space available
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/transformers"
export XDG_CACHE_HOME="/scratch/$USER/.cache"

python -u main.py --data tatoeba --src_lang de --trg_lang ko --attribute input_x_gradient --use_ds tatoeba_filtered
