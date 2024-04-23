#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cpu-preprocessing
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH --time=01:00:00
#SBATCH --output=../slurm_log/output.out
#SBATCH --error=../slurm_log/error.err

module load gcc12-env/12.3.0
module load miniconda3/23.5.2
conda activate my_pytorch_env
cd $WORK/trading_bot

python -m src.model.explore_dataset

jobinfo

