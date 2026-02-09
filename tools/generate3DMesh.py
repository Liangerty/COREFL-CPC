"""
This script generates a 3D mesh for a BL simulation in Plot3D format.
The mesh is saved in the @WORK_DIR directory, and being processed by the readGrid utility for COREFL simulations.
"""

import numpy as np
import os

# Boundary condition labels
BC_IMIN = 4  # inlet
BC_IMAX = 6  # outlet
BC_JMIN = 2  # bottom : slip wall / symmetry
BC_JMAX = 6  # top
BC_KMIN = 8  # front
BC_KMAX = 8  # back
BC_INTERFACE = -1  # block interface boundary condition

# Parameter settings
# x settings, the jet is at x = 0 in [diameter units]
xInlet = 0  # Inlet x-coordinate in [meters]
xOutlet = 0.2  # Core region x-coordinate in [meters] (0.08)
nx = 3001  # Number of grid points in x direction
dx = 400e-6  # Grid size in x direction in [meters] (0.0002)
xBufferNum = 0  # Number of buffer layers in x direction
xBufferGR = 1.1  # Growth rate in x direction
# y settings, as the turbulent boundary layer requires much finer grid, we draw the y grid in a turbulent way no matter the flow is turbulent or laminar.
yFirstLayer = 1e-6  # First layer y grid size in [meters]
yEqualNearWallNum = 10  # Number of equal y grid points near the wall
yGRBL = 1.06  # Growth rate in y direction in boundary layer
yEqualDy = 0.0002  # y grid size in the equal region in [meters]
yEqualHeight = 0.018  # Upper limit of the equal region in [meters]
yBufferNum = 6  # Number of buffer layers in y direction
yBufferGR = 1.1  # Growth rate in y direction
# z settings
dz = 0.0001  # Grid size in z direction in [meters]
zLeft = 0  # Left boundary of the domain in [meters]
zRight = 0.02  # Right boundary of the domain in [meters]

NBLOCK_X = 128  # Number of blocks divided in x direction
NBLOCK_Y = 4  # Number of blocks divided in y direction
FILENAME = "bl.xyz"
WORK_DIR = "512p"


def generate_3D_mesh(
    xLeft=xInlet,
    xRight=xOutlet,
    nx=nx if nx is not None else None,
    dx=dx if nx is None else (xOutlet - xInlet) / (nx - 1),
    xBufferNum=xBufferNum,
    xBufferGR=xBufferGR,
    yFirstLayer=yFirstLayer,
    yEqualNearWallNum=yEqualNearWallNum,
    yGRBL=yGRBL,
    yEqualDy=yEqualDy,
    yEqualHeight=yEqualHeight,
    yBufferNum=yBufferNum,
    yBufferGR=yBufferGR,
    dz=dz,
    zLeft=zLeft,
    zRight=zRight,
    nBlock_x=NBLOCK_X,
    nBlock_y=NBLOCK_Y,
    info=False,
):
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    # Generate full x coordinates first
    if nx is None:
        nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    x = np.linspace(xLeft, xRight, nx)
    # Generate buffer for x
    for i in range(xBufferNum):
        dx = dx * xBufferGR
        xRight = xRight + dx
        x = np.append(x, xRight)
    nx = len(x)
    # print("nx = ", nx)
    if nBlock_x > 1:
        print(f"nx = {nx} + {nBlock_x} - 1 = {nx + nBlock_x - 1}")
    else:
        print(f"nx = {nx}")
    if info:
        np.savetxt(os.path.join(WORK_DIR, "x_full.txt"), x)
    # Calculate points per block (try to make them as equal as possible)
    points_per_block = (nx + nBlock_x - 1) // nBlock_x
    remainder = (nx + nBlock_x - 1) % nBlock_x
    block_sizes = [
        points_per_block + (1 if i < remainder else 0) for i in range(nBlock_x)
    ]

    # Generate y coordinates
    y = [0.0]  # Start from the wall at y = 0
    # Near wall region is equal size
    for i in range(yEqualNearWallNum):
        y.append(y[i] + yFirstLayer)
    # Boundary layer region
    current_y = y[-1]
    current_dy = yFirstLayer
    while current_dy < yEqualDy:
        current_dy *= yGRBL
        current_y += current_dy
        y.append(current_y)
    # Equal region
    current_y = y[-1]
    while current_y < yEqualHeight:
        current_y += yEqualDy
        y.append(current_y)
    # Buffer region
    current_y = y[-1]
    current_dy = yEqualDy
    for _ in range(yBufferNum):
        current_dy *= yBufferGR
        current_y += current_dy
        y.append(current_y)
    y = np.array(y)
    if nBlock_y > 1:
        print(f"ny = {len(y)} + {nBlock_y} - 1 = {len(y) + nBlock_y - 1}")
    else:
        print(f"ny = {len(y)}")
    if info:
        np.savetxt(os.path.join(WORK_DIR, "y_full.txt"), y)

    # divide y into blocks
    points_per_block_y = (len(y) + nBlock_y - 1) // nBlock_y
    remainder_y = (len(y) + nBlock_y - 1) % nBlock_y
    block_sizes_y = [
        points_per_block_y + (1 if i < remainder_y else 0) for i in range(nBlock_y)
    ]

    # Generate z coordinates
    nz = int((zRight - zLeft) / dz) + 1  # Number of grid points in z direction
    z = np.linspace(zLeft, zRight, nz)
    print(f"nz = {len(z)}")
    if info:
        np.savetxt(os.path.join(WORK_DIR, "z_full.txt"), z)

    # Create meshgrid
    blocks = []
    start_idx = 0
    nGrid = 0
    for bsize in block_sizes:
        end_idx = start_idx + bsize
        x_block = x[start_idx:end_idx]

        j_idx = 0
        for j in range(nBlock_y):
            bsize_y = block_sizes_y[j]
            y_block = y[j_idx : j_idx + bsize_y]
            j_idx += bsize_y - 1  # Overlap one point for block interfaces
            X, Y, Z = np.meshgrid(x_block, y_block, z, indexing="ij")
            print(
                f"Block ({len(blocks)}): Shapes:{X.shape}, INDEXES -> x({start_idx}:{end_idx - 1}), y({j_idx - (bsize_y - 1)}:{j_idx}), z(0:{len(z) - 1}), N = {X.size}"
            )
            # print(f"Block ({len(blocks)}): X:{X.shape}, Y:{Y.shape}, Z:{Z.shape}")
            # print the index
            # print(f"x index: {start_idx} to {end_idx - 1}, y index: {j_idx - (bsize_y - 1)} to {j_idx}, z index: 0 to {len(z) - 1}")
            blocks.append((X, Y, Z))
            nGrid += X.size

        start_idx = end_idx - 1  # Overlap one point for block interfaces

    print(f"Total number of grid points: {nGrid}")
    # Write to Plot3D format
    write_plot3d(WORK_DIR, FILENAME, blocks)

    # Write boundary conditions
    bc_filename = FILENAME.replace(".xyz", ".inp")
    write_boundary_conditions(WORK_DIR, bc_filename, blocks, 3)


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

        nBlock_y = NBLOCK_Y

        for b, (nx, ny, nz) in enumerate(block_dims):
            bi = b // nBlock_y
            bj = b % nBlock_y
            block_bc = []
            # i-min face
            if bi == 0:
                block_bc.append((1, 1, 1, ny, 1, nz, BC_IMIN))  # inlet
            else:
                block_bc.append((1, 1, 1, ny, -1, -nz, BC_INTERFACE))  # interface
                # the interface info of the last block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the last block's info
                b_t = (bi - 1) * nBlock_y + bj
                nx_last = block_dims[b_t][0]  # b-1
                block_bc.append(
                    (nx_last, nx_last, 1, ny, -1, -nz, b_t + 1)
                )  # interface

            # i-max face
            if bi == NBLOCK_X - 1:
                block_bc.append((nx, nx, 1, ny, 1, nz, BC_IMAX))  # outlet
            else:
                block_bc.append((nx, nx, 1, ny, -1, -nz, BC_INTERFACE))  # interface
                # the interface info of the next block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the next block's info
                b_t = (bi + 1) * nBlock_y + bj  # 0-based
                block_bc.append((1, 1, 1, ny, -1, -nz, b_t + 1))  # interface

            # j-min face
            if bj == 0:
                block_bc.append((1, nx, 1, 1, 1, nz, BC_JMIN))  # bottom
            else:
                block_bc.append((1, nx, 1, 1, 1, nz, BC_INTERFACE))  # interface
                # the interface info of the last block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the last block's info
                b_t = bi * nBlock_y + (bj - 1)
                ny_last = block_dims[b_t][1]  # b-1
                block_bc.append((1, nx, ny_last, ny_last, 1, nz, b_t + 1))  # interface

            # j-max face
            if bj == nBlock_y - 1:
                block_bc.append((1, nx, ny, ny, 1, nz, BC_JMAX))  # top
            else:
                block_bc.append((1, nx, ny, ny, 1, nz, BC_INTERFACE))  # interface
                # the interface info of the next block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the next block's info
                b_t = bi * nBlock_y + (bj + 1)
                block_bc.append((1, nx, 1, 1, 1, nz, b_t + 1))  # interface

            # Other faces (same for all blocks)
            block_bc.extend(
                [
                    # (1, nx, 1, 1, 1, nz, BC_JMIN),  # bottom
                    # (1, nx, ny, ny, 1, nz, BC_JMAX),  # top
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
    generate_3D_mesh()
    GridgenOrPointwise = 0  # 0 for Gridgen, 1 for Pointwise
    dimension = 3  # 2 for 2D, 3 for 3D
    working_directory = WORK_DIR
    gridFile = os.path.join(WORK_DIR, FILENAME)
    boundaryFile = os.path.join(WORK_DIR, FILENAME.replace(".xyz", ".inp"))
    # gridFile = FILENAME
    # boundaryFile = FILENAME.replace('.xyz', '.inp')
    n_proc = NBLOCK_X * NBLOCK_Y  # number of processes
    isBinary = True  # True for binary, False for ASCII
    writeBinary = False  # True to write binary files, False to write ASCII files
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
