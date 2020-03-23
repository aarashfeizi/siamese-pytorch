#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --job-name=siamese-network-lr00006
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=46G
#SBATCH --time=2-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aarash.feizi@mail.mcgill.ca

source /home/aarash/venv-siamese/bin/activate


python3 train.py -cuda \
        -dsn cub \
        -dsp CUB_200_2011 \
        -sdp images \
        -sp models \
        -gpu 0 \
        -wr 10 \
        -bs 8 \
        -e 50000 \
        -lr 0.00006

