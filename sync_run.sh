#!/bin/bash

#SBATCH --nodelist=steropes # if you need specific nodes
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00 # time requested (D-HH:MM)
#SBATCH --output=slurm.out.txt
#SBATCH --error=slurm.err.txt


# DEST: end with '/'
DEST="/data/$USER/tmp/201204_tmp/"

# sync to destination dir
mkdir -p ${DEST}
rsync -a --delete $(pwd)/ ${DEST}

# move and run script
pushd ${DEST}
bash docker_run.sh
popd

# rsync -a --delete ${DEST}/ $(pwd)/
rsync -a ${DEST} $(pwd)/
