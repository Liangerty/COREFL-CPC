# Examples for COREFL

Some examples are given as test cases for COREFL, one can mimic the input files to set up a new case. The scripts below assumes the corefl executable is in that folder.

For each cases below, the compilation should specify `MAX_SPEC_NUMBER=M`, `MAX_REAC_NUMBER=N`, the two values should be at least `M=1`, `N=1`. If a larger value is needed, we would point that out.

In reality, most cases should be run on a supercomputer, where a script file is a better way to start a simulation. As examples here, we mainly present some small cases that can be run on a personal computer with Nvidia GPUs. The big cases that should be run on more than 1 GPU are explicitly told, and users can modify the mesh generation files to make the grid smaller to test.

## Case specification

### constantVolumeReactor

```bash
./corefl
```

### ODW

Oblique Detonation Wave case. We generate a grid with 4 blocks, and the case should be run with 4 GPUs. You can manually modify the `generate2DMesh.py` to use fewer or more processors and also the grid numbers.

This case should be compiled with `MAX_SPEC_NUMBER=9`, `MAX_REAC_NUMBER=20`.

```bash
python ./generate2DMesh.py
./corefl
```

### reactiveShockTube

This case should be compiled with `MAX_SPEC_NUMBER=9`, `MAX_REAC_NUMBER=18`.

```bash
python ./generate1DMesh.py
./corefl
```

### sinWaveProp

```bash
python ./generate1DMesh.py
./corefl
python ./centerline_error.py
```

### sodShockTube

```bash
python ./generate1DMesh.py
./corefl
```

### STBL

Supersonic Turbulent Boundary Layer case. We generate a grid with 4 blocks, and the case should be run with 4 GPUs. You can manually modify the `generateMeshTBL.py` to use fewer or more processors and also the grid numbers.

```bash
python ./generateMeshTBL.py
mpiexec -n 4 ./corefl
```

## About data treatment

We do not include any post processors in the code. We recommend using the monitor function to monitor the blocks, slices, points that you need, and treat the files after simulations. We are working to publish a post-processor code in the future.
