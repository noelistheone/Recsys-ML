#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=rec
#SBATCH -o output_%j.log   # 所有的标准输出会实时写入这个文件


export HF_HOME=/ocean/projects/cis250095p/xzou3/hf_cache/
export TRANSFORMERS_CACHE=/ocean/projects/cis250095p/xzou3/hf_cache/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250095p/xzou3/hf_cache/datasets
export HF_MODULES_CACHE=/ocean/projects/cis250095p/xzou3/hf_cache/modules
export HF_ASSETS_CACHE=/ocean/projects/cis250095p/xzou3/hf_cache/assets

module load cuda/12.6
module load anaconda3

conda activate /ocean/projects/cis250095p/xzou3/envs/recsys/

# Usage:
#   sbatch job.sh                          # run all configs
#   sbatch job.sh used                     # run only conf/used/
#   sbatch job.sh baseline                 # run only conf/baseline/
#   sbatch job.sh ./conf/used/yelp2018.yaml  # run a single config file

CONFIG_ARG=${1:-"all"}

run_config() {
    echo "======================================"
    echo "Running: $1"
    echo "======================================"
    python main.py --config "$1"
}

if [[ "$CONFIG_ARG" == *.yaml ]]; then
    # Single config file
    run_config "$CONFIG_ARG"
elif [[ "$CONFIG_ARG" == "used" ]]; then
    # Only conf/used/
    for f in ./conf/used/*.yaml; do run_config "$f"; done
elif [[ "$CONFIG_ARG" == "baseline" ]]; then
    # Only conf/baseline/
    for f in ./conf/baseline/*.yaml; do run_config "$f"; done
else
    # All configs
    for f in ./conf/used/*.yaml; do run_config "$f"; done
    for f in ./conf/baseline/*.yaml; do run_config "$f"; done
fi
