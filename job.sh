#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/my_gpu_job.out
#SBATCH --error=logs/my_gpu_job.err
#SBATCH --mail-type=all
#SBATCH --mail-user=your_email@example.com

module purge
module load gcc/11.3.0
module load python/3.9.12
module load py-pip/21.3.1
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6

python cs544_1028.py
