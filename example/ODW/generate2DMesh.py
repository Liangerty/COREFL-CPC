"""
This script generates a 2D mesh for a 1D simulation in Plot3D format.
The mesh is saved in the @WORK_DIR directory, and being processed by the readGrid utility for COREFL simulations.
"""

import numpy as np
import os

# Boundary condition labels
BC_IMIN = 5  # inlet
BC_IMAX = 6  # outlet
BC_JMIN = 3  # bottom : slip wall / symmetry
BC_JMAX = 5  # top
BC_KMIN = 6  # front
BC_KMAX = 6  # back
BC_INTERFACE = -1  # block interface boundary condition

# Parameter settings
# x settings, the jet is at x = 0 in [diameter units]
xInlet = 0  # Inlet x-coordinate in [meters]
xOutlet = 0.024  # Core region x-coordinate in [meters] (0.08)
dx = 5e-6  # Grid size in x direction in [meters] (0.0002)
xBufferNum = 0  # Number of buffer layers in x direction
xBufferGR = 1.1  # Growth rate in x direction
# y settings, as the turbulent boundary layer requires much finer grid, we draw the y grid in a turbulent way no matter the flow is turbulent or laminar.
yBottom = 0  # Bottom y-coordinate in [meters]
yTop = 0.012  # Top y-coordinate in [meters] (0.01)
dy = 5e-6  # First layer y grid size in [meters] (1e-6)
ny = None  # Number of equal y grid points near the wall
yBufferNum = 0  # Number of buffer layers in y direction
yBufferGR = 1.1  # Growth rate in y direction

NBLOCK = 4  # Number of blocks in the mesh
FILENAME = "ODW.xyz"
WORK_DIR = "input"


def generate_1D_mesh(
    xLeft=xInlet,
    xRight=xOutlet,
    dx=dx,
    yBottom=yBottom,
    yTop=yTop,
    dy=dy,
    nBlock=NBLOCK,
    info=False,
):
    # Generate full x coordinates first
    nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    # dx = (xRight - xLeft) / (nx - 1)
    # nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    x = np.linspace(xLeft, xRight, nx)
    # Generate buffer for x
    for i in range(xBufferNum):
      dx = dx * xBufferGR
      xRight = xRight + dx
      x = np.append(x, xRight)
    nx = len(x)

    if info:
        np.savetxt(os.path.join(WORK_DIR, "x_full.txt"), x)
    # Calculate points per block (try to make them as equal as possible)
    points_per_block = nx // nBlock
    remainder = nx % nBlock
    block_sizes = [
        points_per_block + (1 if i < remainder else 0) for i in range(nBlock)
    ]

    # Generate y coordinates
    Ly = yTop - yBottom  # Total length in y direction
    y = np.linspace(yBottom, yTop, int(Ly / dy) + 1)

    # Generate buffer for y
    for i in range(yBufferNum):
        dy = dy * yBufferGR
        yTop = yTop + dy
        y = np.append(y, yTop)

    if info:
        np.savetxt(os.path.join(WORK_DIR, "y_full.txt"), y)
    # Generate z coordinates
    z = np.array([0])  # Single layer in z direction
    if info:
        np.savetxt(os.path.join(WORK_DIR, "z_full.txt"), z)
    # Create meshgrid
    blocks = []
    start_idx = 0
    for bsize in block_sizes:
        end_idx = start_idx + bsize
        x_block = x[start_idx:end_idx]
        X, Y, Z = np.meshgrid(x_block, y, z, indexing="ij")
        blocks.append((X, Y, Z))
        start_idx = end_idx - 1  # Overlap one point for block interfaces

    # Write to Plot3D format
    write_plot3d(WORK_DIR, FILENAME, blocks)
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # # Write to Plot3D format
    # with open(os.path.join(WORK_DIR, FILENAME), 'wb') as f:
    #     # Write number of blocks
    #     np.array([1], dtype=np.int32).tofile(f)
    #     # Write dimensions
    #     dims = np.array([X.shape[0], X.shape[1], X.shape[2]], dtype=np.int32)
    #     dims.tofile(f)
    #     # Write coordinates
    #     X.ravel(order='F').tofile(f)
    #     Y.ravel(order='F').tofile(f)
    #     Z.ravel(order='F').tofile(f)

    # Write boundary conditions
    bc_filename = FILENAME.replace(".xyz", ".inp")
    write_boundary_conditions(WORK_DIR, bc_filename, blocks, 2)


def write_plot3d(workdir, filename, blocks):
    # the files should be output to a folder "readGrid"
    # First, create the directory if it doesn't exist
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    with open(os.path.join(workdir, filename), "wb") as f:
        # Write number of blocks
        np.array([len(blocks)], dtype=np.int32).tofile(f)
        dimensions = np.array([X.shape for X, _, _ in blocks], dtype=np.int32)
        dimensions.tofile(f)

        # Write coordinates for all blocks
        for X, Y, Z in blocks:
            # dims = np.array(X.shape, dtype=np.int32)
            # dims.tofile(f)
            X.ravel(order="F").tofile(f)
            Y.ravel(order="F").tofile(f)
            Z.ravel(order="F").tofile(f)


def write_boundary_conditions(workdir, filename, blocks, dims=3):
    # if dims == 2:
    #     with open(os.path.join(WORK_DIR, filename), 'w') as f:
    #         f.write(f"1\n")
    #         # Write number of blocks
    #         f.write(f"1\n")
    #         # Write dimensions
    #         nx = blocks[0][0].shape[0]
    #         ny = blocks[0][0].shape[1]
    #         f.write(f"{nx}\t{ny}\t1\nblock-0\n4\n")
    #         # Write boundary conditions
    #         f.write(f"1\t1\t1\t{ny}\t{BC_IMIN}\n")  # inlet
    #         f.write(f"{nx}\t{nx}\t1\t{ny}\t{BC_IMAX}\n") # outlet
    #         f.write(f"1\t{nx}\t1\t1\t{BC_JMIN}\n")
    #         f.write(f"1\t{nx}\t{ny}\t{ny}\t{BC_JMAX}\n")

    #     return
    if dims == 2:
        block_dims = [X.shape for X, _, _ in blocks]
        bc_data = []

        for b, (nx, ny, nz) in enumerate(block_dims):
            block_bc = []
            # i-min face
            if b == 0:
                block_bc.append((1, 1, 1, ny, BC_IMIN))  # inlet
            else:
                block_bc.append((1, 1, 1, ny, BC_INTERFACE))  # interface
                # the interface info of the last block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the last block's info
                nx_last = block_dims[b - 1][0]
                block_bc.append((nx_last, nx_last, 1, ny, b))  # interface

            # i-max face
            if b == len(blocks) - 1:
                block_bc.append((nx, nx, 1, ny, BC_IMAX))  # outlet
            else:
                block_bc.append((nx, nx, 1, ny, BC_INTERFACE))  # interface
                # the interface info of the next block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the next block's info
                block_bc.append((1, 1, 1, ny, b + 2))

            # Other faces (same for all blocks)
            block_bc.extend(
                [
                    (1, nx, 1, 1, BC_JMIN),  # bottom
                    (1, nx, ny, ny, BC_JMAX),  # top
                ]
            )
            bc_data.append(block_bc)

        if not os.path.exists(workdir):
            os.makedirs(workdir)
        with open(os.path.join(workdir, filename), "w") as f:
            f.write(f"1\n")
            # Write number of blocks
            f.write(f"{len(blocks)}\n")

            # For each block
            for block_id, (dims, bcs) in enumerate(zip(block_dims, bc_data)):
                nx, ny, nz = dims
                f.write(f"{nx}\t{ny}\t{nz}\nblock-{block_id}\n4\n")
                for bc in bcs:
                    f.write(" ".join(map(lambda x: f"{x:6d}", bc)) + "\n")
        return
    else:  # 3d case
        block_dims = [X.shape for X, _, _ in blocks]
        bc_data = []

        for b, (nx, ny, nz) in enumerate(block_dims):
            block_bc = []
            # i-min face
            if b == 0:
                block_bc.append((1, 1, 1, ny, 1, nz, BC_IMIN))  # inlet
            else:
                block_bc.append((1, 1, 1, ny, -1, -nz, BC_INTERFACE))  # interface
                # the interface info of the last block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the last block's info
                nx_last = block_dims[b - 1][0]
                block_bc.append((nx_last, nx_last, 1, ny, -1, -nz, b))  # interface

            # i-max face
            if b == len(blocks) - 1:
                block_bc.append((nx, nx, 1, ny, 1, nz, BC_IMAX))  # outlet
            else:
                block_bc.append((nx, nx, 1, ny, -1, -nz, BC_INTERFACE))  # interface
                # the interface info of the next block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the next block's info
                block_bc.append((1, 1, 1, ny, -1, -nz, b + 2))

            # Other faces (same for all blocks)
            block_bc.extend(
                [
                    (1, nx, 1, 1, 1, nz, BC_JMIN),  # bottom
                    (1, nx, ny, ny, 1, nz, BC_JMAX),  # top
                    (1, nx, 1, ny, 1, 1, BC_KMIN),  # front
                    (1, nx, 1, ny, nz, nz, BC_KMAX),  # back
                ]
            )
            bc_data.append(block_bc)

        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
        with open(os.path.join(WORK_DIR, filename), "w") as f:
            f.write(f"1\n")
            # Write number of blocks
            f.write(f"{len(blocks)}\n")

            # For each block
            for block_id, (dims, bcs) in enumerate(zip(block_dims, bc_data)):
                nx, ny, nz = dims
                f.write(f"{nx}\t{ny}\t{nz}\nblock-{block_id}\n6\n")
                for bc in bcs:
                    f.write(" ".join(map(lambda x: f"{x:6d}", bc)) + "\n")


def main():
    # Generate the multiblock mesh
    generate_1D_mesh()
    GridgenOrPointwise = 0  # 0 for Gridgen, 1 for Pointwise
    dimension = 2  # 2 for 2D, 3 for 3D
    working_directory = WORK_DIR
    gridFile = os.path.join(WORK_DIR, FILENAME)
    boundaryFile = os.path.join(WORK_DIR, FILENAME.replace(".xyz", ".inp"))
    # gridFile = FILENAME
    # boundaryFile = FILENAME.replace('.xyz', '.inp')
    n_proc = NBLOCK
    isBinary = True  # True for binary, False for ASCII
    writeBinary = True  # True to write binary files, False to write ASCII files
    setZ = True  # True to set z coordinate to zero for 2D mesh
    zValue = 0.0  # z coordinate value for 2D mesh
    from read_grid import read_grid

    read_grid(
        GridgenOrPointwise,
        dimension,
        gridFile,
        boundaryFile,
        n_proc,
        isBinary,
        writeBinary,
        setZ,
        zValue,
        working_directory,
    )


if __name__ == "__main__":
    main()
