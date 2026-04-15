#!/bin/bash
#SBATCH --job-name=PlaylistGenreClassification  # Job name
#SBATCH --output=output_%j.txt       # Output file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.txt         # Error file (%j will be replaced with the job ID)
#SBATCH --time=1-0:0                 # Time limit (DD-HH:MM)
#SBATCH --partition=teaching         # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

# use sbatch --export=NUM_AUGMENTATIONS=x training.sh 
# with x being the target number
# echo $NUM_AUGMENTATIONS
# same for ESTIMATE_PATH with the path to estimate.json
# echo $ESTIMATE_PATH

/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/.venv/bin/python /home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/PlaylistGenreClassification/CLAP_model.py