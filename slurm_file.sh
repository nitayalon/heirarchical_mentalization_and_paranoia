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
#SBATCH --cpus-per-task=4
#SBATCH -a 10-999:50%20
#SBATCH —exclusive=user


#SBATCH --mail-type=END,FAIL
## *** YOU NEED TO FILL IN YOUR KYB EMAIL ADDRESS HERE ***
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
# Wall clock limit:
#SBATCH --time=1-12:30

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

ENV=first_task
SOFTMAX_TEMP=0.01
RECEIVER_TOM=DoM2
SENDER_TOM=DoM1

echo "Simulating with seed $SLURM_ARRAY_TASK_ID"
time singularity exec ${CONTAINER_PATH} python main.py  --environment $ENV --seed $SLURM_ARRAY_TASK_ID --softmax_temp $SOFTMAX_TEMP --sender_tom $SENDER_TOM --receiver_tom  $RECEIVER_TOM 
