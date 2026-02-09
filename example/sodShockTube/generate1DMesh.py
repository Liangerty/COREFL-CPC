'''
This script generates a 2D mesh for a 1D simulation in Plot3D format.
The mesh is saved in the @WORK_DIR directory, and being processed by the readGrid utility for COREFL simulations.
'''

import numpy as np
import os

# Boundary condition labels
BC_IMIN = 6   # inlet
BC_IMAX = 6   # outlet
BC_JMIN = 6   # bottom
BC_JMAX = 6   # top
BC_KMIN = 6   # front
BC_KMAX = 6   # back
BC_INTERFACE = -1  # block interface boundary condition

# Parameter settings
# x settings, the jet is at x = 0 in [diameter units]
xInlet = 0  # Inlet x-coordinate in [meters]
xOutlet = 1  # Core region x-coordinate in [meters] (0.08)
nx = 201  # Number of grid points in x direction, nx and dx will be used to calculate each other, only one of them needs to be set, the other one can be set to None
dx = 5e-3  # Grid size in x direction in [meters] (0.0002)
# y settings, as the turbulent boundary layer requires much finer grid, we draw the y grid in a turbulent way no matter the flow is turbulent or laminar.
dy = 0.01 # First layer y grid size in [meters] (1e-6)
ny = 7 # Number of equal y grid points near the wall
FILENAME = "shockTube.xyz"
WORK_DIR = "input"

def generate_1D_mesh(xLeft=xInlet, xRight=xOutlet, dx = dx if nx is None else None, nx = nx, dy=dy, ny=ny, info=False):
    # Generate full x coordinates first
    if dx is not None:
        nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    else:
        dx = (xRight - xLeft) / (nx - 1)  # Grid size in x direction
    # dx = (xRight - xLeft) / (nx - 1)
    # nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    x = np.linspace(xLeft, xRight, nx)
    if info:
      np.savetxt(os.path.join(WORK_DIR, "x_full.txt"), x)

    # Generate y coordinates
    Ly = dy * (ny - 1)  # Total length in y direction
    y = np.linspace(-0.5*Ly, 0.5*Ly, ny)
    if info:
      np.savetxt(os.path.join(WORK_DIR, "y_full.txt"), y)
    # Generate z coordinates
    z = np.array([0])  # Single layer in z direction
    if info:
      np.savetxt(os.path.join(WORK_DIR, "z_full.txt"), z)
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Write to Plot3D format
    with open(os.path.join(WORK_DIR, FILENAME), 'wb') as f:
        # Write number of blocks
        np.array([1], dtype=np.int32).tofile(f)
        # Write dimensions
        dims = np.array([X.shape[0], X.shape[1], X.shape[2]], dtype=np.int32)
        dims.tofile(f)
        # Write coordinates
        X.ravel(order='F').tofile(f)
        Y.ravel(order='F').tofile(f)
        Z.ravel(order='F').tofile(f)

    # Write boundary conditions
    bc_filename = FILENAME.replace('.xyz', '.inp')
    write_boundary_conditions(bc_filename, [(X, Y, Z)],2)

def write_boundary_conditions(filename, blocks, dims=3):
    if dims == 2:
        with open(os.path.join(WORK_DIR, filename), 'w') as f:
            f.write(f"1\n")
            # Write number of blocks
            f.write(f"1\n")
            # Write dimensions
            nx = blocks[0][0].shape[0]
            ny = blocks[0][0].shape[1]
            f.write(f"{nx}\t{ny}\t1\nblock-0\n4\n")
            # Write boundary conditions
            f.write(f"1\t1\t1\t{ny}\t{BC_IMIN}\n")  # inlet
            f.write(f"{nx}\t{nx}\t1\t{ny}\t{BC_IMAX}\n") # outlet
            f.write(f"1\t{nx}\t1\t1\t{BC_JMIN}\n")
            f.write(f"1\t{nx}\t{ny}\t{ny}\t{BC_JMAX}\n")
        
        return
    else: # 3d case
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
                nx_last = block_dims[b-1][0]
                block_bc.append((nx_last, nx_last, 1, ny, -1, -nz, b))  # interface


            # i-max face
            if b == len(blocks) - 1:
                block_bc.append((nx, nx, 1, ny, 1, nz, BC_IMAX))  # outlet
            else:
                block_bc.append((nx, nx, 1, ny, -1, -nz, BC_INTERFACE))  # interface
                # the interface info of the next block is written in the next line which communicates with this face
                # the info of nx/ny/nz is the next block's info
                block_bc.append((1, 1, 1, ny, -1, -nz, b+2))
            
            # Other faces (same for all blocks)
            block_bc.extend([
                (1, nx, 1, 1, 1, nz, BC_JMIN),      # bottom
                (1, nx, ny, ny, 1, nz, BC_JMAX),    # top
                (1, nx, 1, ny, 1, 1, BC_KMIN),      # front
                (1, nx, 1, ny, nz, nz, BC_KMAX)     # back
            ])
            bc_data.append(block_bc)
        
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
        with open(os.path.join(WORK_DIR, filename), 'w') as f:
            f.write(f"1\n");
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
    boundaryFile = os.path.join(WORK_DIR, FILENAME.replace('.xyz', '.inp'))
    # gridFile = FILENAME
    # boundaryFile = FILENAME.replace('.xyz', '.inp')
    n_proc = 1
    isBinary = True  # True for binary, False for ASCII
    writeBinary = True  # True to write binary files, False to write ASCII files
    setZ = True  # True to set z coordinate to zero for 2D mesh
    zValue = 0.0  # z coordinate value for 2D mesh
    from read_grid import read_grid
    read_grid(GridgenOrPointwise, dimension, gridFile, boundaryFile, n_proc, isBinary, writeBinary, setZ, zValue, working_directory)

if __name__ == "__main__":
    main() 