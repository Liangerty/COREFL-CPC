# Use **readGrid** to prepare the grid file

## About the **readGrid** tool

COREFL needs to read the computational mesh in a fixed format, and **readGrid** supplies such function. By reading a mesh given in `Plot3D` format, it generates the files for the simulation.

The **readGrid** tool will read the "`gridFile.dat`" and "`gridFile.inp`", where "`gridFile`" is the filename, and output two folders `grid` and `boundary_condition`, which should be moved to the `input` folder and read by COREFL.

The advantage of using such a file is that the corresponding processes need only read their own blocks instead of waiting for asigning. But this may be improved or integrated in COREFL in the future.

The tool is in the folder **readGrid**. The compilation is straightforward and is omitted here. This code can be compiled on Windows systems. A pre-compiled executable is also given in the folder if anyone uses this tool on Windows before uploading the grid to linux clusters (which is what we do).

## Use **readGrid**

The grid for COREFL is structured grid in `Plot3D` format. The `Plot3D` files including `name.dat` and `name.inp` contain the grid coordinates and boundary condition information, respectively.

> You can generate the grid with a commercial software supporting `Plot3D` or by writing codes. For example, **Pointwise** and **Gridgen** both support this file format. In **Pointwise**, you choose the solver "Gridgen Generic", and by exporting CAE, you get those two files.

> Note that, we do not have automatic blocking in COREFL. You need to partition the blocks manually when generating grid in softwares if a parallel computation is needed.

The tool gets the information from a file named `input.txt`, where we need to put the following info into it:

```c++
// Gridgen(0) or Pointwise(1)
int GridgenOrPointwise = 1

// Dimension
int dimension = 3
// grid file name
string gridFile = combustor.dat
// boundary file name
string boundaryFile = combustor.inp
// number of processors to use
int n_proc = 2
// if the grid is in ASCII(0) or in binary(1) form
bool isBinary = 1
// if we write the grid in ASCII(0) or in binary(1) form
int writeBinary = 1

// If 2D, set all Z values to the given value
bool setZ = 0
real zValue = 0.0
```

With the two files, the tool **readGrid** would generate two folders called `grid` and `boundary_condition`, these should be put into the `input` folder in the work directory.
