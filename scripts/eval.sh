#!/bin/bash

module load jq/1.5

NUM_WORKERS=4
TIME=24:00:00
MEM_PER_CPU=10240
MEM_PER_CPU_MNIST=25G
NTASKS_PER_NODE=20
CPUS_PER_TASK=16
GPUS="rtx_2080_ti"

json_file="/cluster/project/sachan/callen/results_ICLR/final_models.json"


LR=1e-4
EPOCH=1
BATCH=128

DATASETs=("mnist" "fashionmnist" "cifar10" "celeba" ) #"tinyimagenet") 
SELF_SUPs=(1) #0 2 10 3 31
SEEDs=(1234)

for DATASET in "${DATASETs[@]}"
do
for SELF_SUP in "${SELF_SUPs[@]}"
do
for SEED in "${SEEDs[@]}"
do

if [ $DATASET == "cifar10" ]; then
LAYERS=R18
ZDIM=64
CLASSES=10
NUM_WORKERS2=$NUM_WORKERS
fi

if [ $DATASET == "celeba" ]; then
LAYERS=R18
ZDIM=64
CLASSES=5
BATCH=32
NUM_WORKERS2=0
#GPUS="rtx_3090"
fi

if [ $DATASET == "mnist" ]; then
LAYERS=3
ZDIM=10
CLASSES=10
NUM_WORKERS2=$NUM_WORKERS
fi

if [ $DATASET == "fashionmnist" ]; then
LAYERS=3
ZDIM=10
CLASSES=10
NUM_WORKERS2=$NUM_WORKERS
fi

ENVIRONMENT_LEVEL1=".$DATASET"
ENVIRONMENT_LEVEL2=".[\"$SELF_SUP\"]"
ENVIRONMENT_LEVEL3=".[\"$SEED\"]"

dataset_entries=$(jq -r $ENVIRONMENT_LEVEL1 "$json_file")
model_entries=$(echo "$dataset_entries" | jq -r $ENVIRONMENT_LEVEL2)
CHECKPOINT=$(echo "$model_entries" | jq -r $ENVIRONMENT_LEVEL3)

if [ -n "$CHECKPOINT" ]; then
JOB="python ./main_inference.py -B "$BATCH" -H "$CHECKPOINT" -w "$NUM_WORKERS2" -d "$DATASET" -e "$EPOCH" -l "$LAYERS" -L "$LR" -Z "$ZDIM" -s "$SEED"" 
NAME="/cluster/project/sachan/callen/results_ICLR/generation_"$DATASET"/output_"$DATASET"_"$ZDIM"_"$LR"_"$LAYERS"_"$EPOCH"_"$SELF_SUP"_"$SEED""
echo "$JOB"
sbatch -o "$NAME" -n "$NUM_WORKERS" --time="$TIME" --mem-per-cpu="$MEM_PER_CPU_MNIST" -p gpu --gpus="$GPUS":1 --wrap="nvidia-smi;$JOB"
fi
done; done; done;


