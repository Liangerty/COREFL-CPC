# Tools folder

Several tools are given here, and more are to be added in the future.

## generate3DMesh.py

This file is a script to generate a mesh for supersonic turbulent boundary layer simulations, and one can modify the values to get a mesh with your wanted size. The mesh can be split into multiple blocks in `x` or `y` directions, as you specify. The script would call the read_grid function after generating files, and the files that can be read by COREFL are generated afterwards.

## read_grid.py

This file converts the Plot3D format files into the COREFL-readable files.

## TecplotUtilsGXL.py

This file reads the Tecplot `plt` file output by COREFL. One can read and understand this file (Let AI help you) and use it to write a post-processing code.
