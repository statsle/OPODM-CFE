#!/bin/bash

#SBATCH --job-name=...

#SBATCH --ntasks=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=1-00:00

#SBATCH --output=out/o%j.out
#SBATCH --error=out/e%j.err

#SBATCH --account=...


day="$(date +"%d-%m-%Y")"
WORKDIR=test_runs/$day-$SLURM_JOB_ID
mkdir -p "$WORKDIR"

exec >"$WORKDIR/stdout.log" 2>"$WORKDIR/stderr.log"

module load cuda/12.2
module load python/3.11.5
source $HOME/venv/bin/activate

export HF_HOME=...

export OPM_NUM_THREADS=1 


export SCRIPT=".../test.py"
export SCRIPT_ARGS=" \
    --write_dir ... \
    --model_dir ... \
    --test_mode vs_model \
    --seed 99 \
    --mixed_precision fp16 \
    --prompts_path ... \
    --second_model ... \
    --win_threshold 0.5 \
    --max_prompts ... \
    "

echo "Environment ready, starting test..."

python $SCRIPT $SCRIPT_ARGS

deactivate
