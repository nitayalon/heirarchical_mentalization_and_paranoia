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
#SBATCH --exclusive=user


#SBATCH --mail-type=END,FAIL
## *** YOU NEED TO FILL IN YOUR KYB EMAIL ADDRESS HERE ***
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
# Wall clock limit:
#SBATCH --time=1-12:30

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

GAME=G_1
SOFTMAX_TEMP=0.1
DURATION=12
Aleph_Ipomdp=False
DELTA=1.1

echo "Simulating with seed $SLURM_ARRAY_TASK_ID"
time singularity exec ${CONTAINER_PATH} python zero_sum_game/zero_sum_game_task.py  --payout_matrix $GAME --seed $SLURM_ARRAY_TASK_ID --softmax_temp $SOFTMAX_TEMP --duration $DURATION --aleph_ipomdp $Aleph_Ipomdp --strong_typicality_delta $DELTA 

