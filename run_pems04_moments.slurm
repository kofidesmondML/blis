#!/bin/bash
#SBATCH -J blis_PEMS04
#SBATCH -o /bsuscratch/desmondboateng/blis/logs/%x-%j.out
#SBATCH -e /bsuscratch/desmondboateng/blis/logs/%x-%j.err
#SBATCH -p bsudfq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00

set -eo pipefail

cd /bsuscratch/desmondboateng/blis

source ~/miniforge3/etc/profile.d/conda.sh
conda activate blis

echo "Computing BLIS scattering moments for PEMS04"
echo "Date: $(date)"
echo "Python: $(which python)"

python scripts/calculate_scattering.py \
    --scattering_type blis \
    --wavelet_type W2 \
    --largest_scale 4 \
    --highest_moment 3 \
    --dataset traffic \
    --sub_dataset PEMS04

echo "Finished PEMS04"
echo "Date: $(date)"
