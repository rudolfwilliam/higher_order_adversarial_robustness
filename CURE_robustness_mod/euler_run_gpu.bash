#!/bin/bash

# Change module environment
env2lmod

# Load required modules
module load gcc/6.3.0 python_gpu/3.7.4

# Activate the virtual environment
source .venv/bin/activate

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1

# Submit the program
echo "You will be notified via your ETH e-mail when the execution has started and finished"
bsub -W 24:00 -n 1 -R rusage[mem=32000,scratch=1000] -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" -N -B -o logs/log_$DATETIME.txt python reproduce_results.py
