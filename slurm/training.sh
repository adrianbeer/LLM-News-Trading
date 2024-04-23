#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=nn_training
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000
#SBATCH --time=10:00:00
#SBATCH --output=../slurm_log/train_output.out
#SBATCH --error=../slurm_log/train_error.err
#SBATCH --partition=gpu

module load gcc12-env/12.3.0
module load miniconda3/23.5.2
conda activate my_pytorch_env
cd $WORK/trading_bot

python -m src.model.training \
	--batch_size 16 \
	--hidden_layer_size 256 \
	--learning_rate 0.00005 \
	--epochs 5 \
	--dropout_rate 0.1 \
	--task "classification" 
