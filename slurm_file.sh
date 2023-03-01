#!/bin/bash -l
#SBATCH -o ./slurm_logs/%x_%j_tjob.out
#SBATCH -e ./slurm_logs/%x_%j_tjob.err
# Initial working directory:
#SBATCH -D ./
#
# Queue (Partition):
#SBATCH --partition=compute # nyx partitions: compute, highmem, gpu
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive=user
#
#SBATCH --mail-type=END,FAIL, BEGIN
## *** YOU NEED TO FILL IN YOUR KYB EMAIL ADDRESS HERE ***
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#
# Wall clock limit:
#SBATCH --time=1-12:30
omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

ENV=basic_task
SOFTMAX_TEMP=0.5
AGENT_TOM=DoM0
SUBJECT_TOM=DoM-1

echo "Simulating with seed $SLURM_ARRAY_TASK_ID"
time singularity exec ${CONTAINER_PATH} python main.py  --environment $ENV --seed $SLURM_ARRAY_TASK_ID --softmax_temp $SOFTMAX_TEMP --agent_tom $AGENT_TOM --subject_tom $SUBJECT_TOM