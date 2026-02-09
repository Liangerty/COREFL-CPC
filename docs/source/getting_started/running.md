# Run a case

A typical folder for running COREFL is as follows:

- input/
  - grid/*
  - boundary_condition/*
  - chemistry/*
  - setup.txt
- run.sh
- corefl(optional)

All settings are set in the `setup.txt` file in the following manner:

`type name = value`

There should be space between each two symbols.

The grid information is included via the `grid` and `boundary_condition` folders, which is acquired by a `readGrid` tool described in {doc}`readGrid <readGrid>`

To run COREFL on clusters, we need a script file to run it. The executable is optional because we can specify the path to the exectuble in the script instead.

In our environment, we have 4 Nvidia A100s on a node, and 4 nodes. We can write the script as follows:

```bash
#!/bin/bash
#SBATCH --job-name=case1    # name of the job
#SBATCH --nodes=2           # number of nodes to use
#SBATCH --ntasks-per-node=4 # number of tasks per nodes
#SBATCH --gres=gpu:4        # number of GPUs per nodes
#SBATCH --qos=gpugpu        # included when more than 1 node is used

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

mpirun -n 8 \ # total number of processes to be started
  --mca btl tcp,self \
  --mca btl_tcp_if_include eth0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile ${HOSTFILE} \
  /path/to/corefl
```

With the above script, the corefl will be started. An `output` folder will be created which contains all output files.

In the output folder, a file named `flowfield.plt` will be created, which is the instantaneous flowfield file. A folder called `message` will also be created. The flowfield file, and the message folder, are necessary for starting a computation with existing results.
