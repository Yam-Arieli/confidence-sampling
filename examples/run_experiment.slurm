#!/bin/bash
#SBATCH --job-name=experiment_trainset
#SBATCH --output=experiment_output.txt
#SBATCH --error=experiment_error.txt
#SBATCH --gres=gpu:a5000         # ✅ Use specific GPU type
#SBATCH --mem=64G                  # System RAM
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00            # Max runtime
#SBATCH --partition=short         # Make sure this matches an available partition

# Activate your environment
source place_holder/bin/activate

# Run your inference script
python3 place_holder.py \
  --config_experiment_file "$1" 