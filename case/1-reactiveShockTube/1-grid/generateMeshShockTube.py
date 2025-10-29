import numpy as np
import os

# Boundary condition labels
BC_IMIN = 2  # inlet
BC_IMAX = 6  # outlet
BC_JMIN = 6  # bottom
BC_JMAX = 6  # top
BC_KMIN = 6  # front
BC_KMAX = 6  # back
BC_INTERFACE = -1  # block interface boundary condition

# Parameter settings
# x settings, the jet is at x = 0 in [diameter units]
xInlet = 0  # Inlet x-coordinate in [meters]
xOutlet = 0.12  # Core region x-coordinate in [meters] (0.08)
dx=1e-4  # Grid size in x direction in [meters] (0.0002)
# y settings, as the turbulent boundary layer requires much finer grid, we draw the y grid in a turbulent way no matter the flow is turbulent or laminar.
dy = 0.01 # First layer y grid size in [meters] (1e-6)
ny = 7 # Number of equal y grid points
# z settings
# zCoreLeft = -0.03 # Left boundary of the core region in [meters]
# zCoreRight = 0.03 # Right boundary of the core region in [meters]
# dz_min= 0.0001 # Grid size in z direction in jet ranges in [meters]
# dz = 0.00025 # Grid size in z direction in [meters]
# zLeft = -0.05 # Left boundary of the domain in [meters]
# zRight = 0.05 # Right boundary of the domain in [meters]
# zGR = 1.05 # Growth rate in z direction
FILENAME = "shockTube.xyz"
# multi block settings
# x_split = 4  # Number of blocks to split in x direction

def generate_shockTube_mesh(xLeft=xInlet, xRight=xOutlet, dx=dx, dy=dy, ny=ny):
    # Generate full x coordinates first
    # dx = (xRight - xLeft) / (nx - 1)
    nx = int((xRight - xLeft) / dx) + 1  # Number of grid points in x direction
    x = np.linspace(xLeft, xRight, nx)
    np.savetxt("x_full.txt", x)

    # Generate y coordinates
    Ly = dy * (ny - 1)  # Total length in y direction
    y = np.linspace(-0.5*Ly, 0.5*Ly, ny)
    np.savetxt("y_full.txt", y)
    # Generate z coordinates
    z = np.array([0])  # Single layer in z direction
    np.savetxt("z_full.txt", z)
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Write to Plot3D format
    if not os.path.exists("readGrid"):
        os.makedirs("readGrid")
    with open(os.path.join("readGrid", FILENAME), 'wb') as f:
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
        with open(os.path.join("readGrid", filename), 'w') as f:
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
        
        if not os.path.exists("readGrid"):
            os.makedirs("readGrid")
        with open(os.path.join("readGrid", filename), 'w') as f:
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
    generate_shockTube_mesh()

if __name__ == "__main__":
    main() 