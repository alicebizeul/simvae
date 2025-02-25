#!/bin/bash

# setup environment 
conda activate simvae

# ressource specs
NUM_WORKERS=4
TIME=24:00:00
MEM_PER_CPU=4G
MEM_PER_GPU=12G

# path
DATA_PATH="/cluster/scratch/abizeul"
OUTPUT_PATH="/cluster/home/abizeul/simvae/"

# hyperparameters
AUG=2
DATASETs=("mnist" "fashionmnist") 
PRIORs=("gaussian" )

# loop over hyperparameters
for PRIOR in "${PRIORs[@]}"
do
for DATASET in "${DATASETs[@]}"
do

if [ $DATASET == "cifar10" ]; then
LAYERS=R18
ZDIM=64
CLASSES=10
fi

if [ $DATASET == "celeba" ]; then
LAYERS=R18
ZDIM=64
CLASSES=5
GPUS="rtx_3090"
TIME=24:00:00
fi

if [ $DATASET == "mnist" ]; then
LAYERS=3
ZDIM=10
CLASSES=10
fi

if [ $DATASET == "fashionmnist" ]; then
LAYERS=3
ZDIM=10
CLASSES=10
fi

JOB="python main.py -c "$CLASSES" -a "$AUG" -w "$NUM_WORKERS" -d "$DATASET" -l "$LAYERS" -Z "$ZDIM" -p "$OUTPUT_PATH" -P "$DATA_PATH" -o "$PRIOR"" 
NAME="../simvae_output_"$DATASET"_"$ZDIM"_"$LAYERS"_"$AUG"_"$PRIOR""
echo "$JOB"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --time="$TIME" --mem-per-cpu="$MEM_PER_CPU" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;


