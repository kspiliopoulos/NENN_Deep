#!/bin/bash -l

# Set SCC project
#$ -P scalingnn

# Specify hard time limit for the job
#   The job will be aborted if it runs longer than this time
#   The default time is 12 hours
#$ -l h_rt=24:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent)
#$ -m ea

# Overwrites the default email address used to send the job report
#$ -M jyu32@bu.edu

# Give job a name
#$ -N scalingnn

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o scalingnn.qlog

# Specify the number of requested CPUs
#$ -pe omp 1

# Specify the number of requested GPUs
#$ -l gpus=1

# Specify minimum GPU capability. Current choices for CAPABILITY are 2.0, 3.5, 6.0, and 7.0.
#$ -l gpu_c=6.0

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "dataset=$1, model=$2, criterion=$3"
echo "gI=$4, gII=$5, gIII=$6, hI=$7, hII=$8, hIII=$9, e=${10}, b=${11}"
echo "dir=${12}"
echo "=========================================================="

# Load modules
module load python3/3.7.7
module load pytorch/1.6.0


python process.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}
