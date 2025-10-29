#!/bin/bash
#SBATCH --job-name=MLComb   # 作业名称
#SBATCH --nodes=1           # 请求两个节点
#SBATCH --ntasks-per-node=1 # 每个节点启动的任务数
#SBATCH --gres=gpu:1        # 每个节点请求的GPU数
#SBATCH --qos=gpugpu        # 跨界点运行必须有的命令

module purge
module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

### Job ID
JOB_ID="${SLURM_JOB_ID}"
### hosfile
HOSTFILE="hostfile.${JOB_ID}"
GPUS=4

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

mpirun -n 1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_include eth0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile ${HOSTFILE} \
  /home/bingxing2/home/scx6d0j/GuoXL/code/corefl-cpc/corefl
