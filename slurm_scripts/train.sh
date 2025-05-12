#!/bin/bash

#SBATCH --job-name=...
#SBATCH --output=out/o%j.out
#SBATCH --error=out/e%j.err
#SBATCH --nodes=1
#SBATCH --mem=25G
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --time=4-00:00              # maximum execution time (HH:MM:SS)

#SBATCH --account=...

######################
### Set enviroment ###
######################
day="$(date +"%d-%m-%Y")"
WORKDIR=sd_runs/$day-$SLURM_JOB_ID
mkdir -p "$WORKDIR"

exec >"$WORKDIR/stdout.log" 2>"$WORKDIR/stderr.log"

module load cuda/12.2
module load python/3.11.5
source $HOME/venv/bin/activate

export HF_HOME=...
export WANDB_MODE=...
export WANDB_API_KEY=...
export GPUS_PER_NODE=4
######################

export SCRIPT=".../main.py"
export SCRIPT_ARGS=" \
    --allow_tf32 \
    --config .../config.yaml \
    --mixed_precision fp16 \
    "

echo "Environment ready, accelerate launch starting..."

accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS

deactivate
