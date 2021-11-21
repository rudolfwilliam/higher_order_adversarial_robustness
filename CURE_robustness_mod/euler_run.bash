#!/bin/bash
module load python/3.6.0
module load python/3.7.1

# Activate the virtual environment
source .venv/bin/activate

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1

# Submit the program
echo "You will be notified via your ETH e-mail when the execution has started and finished"
bsub -W 24:00 -n 1 -R rusage[mem=64000,scratch=1000] -N -B -o logs/log_$DATETIME.txt python reproduce_results.py
