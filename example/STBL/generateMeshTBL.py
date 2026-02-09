import numpy as np
import os

# Boundary condition labels
BC_IMIN = 5  # inlet
BC_IMAX = 6  # outlet
BC_JMIN = 2  # bottom
BC_JMAX = 6  # top
BC_KMIN = 8  # front
BC_KMAX = 8  # back
BC_INTERFACE = -1  # block interface boundary condition

# Parameter settings
# x settings, the parameters are set according to (Pirozzoli et al. 2004), there is no smoothness operation for the dx from transition part to turbulent part, which may cause some problems at the initial turbulent part. But as we concentrate on the statistics of the turbulent part at the end of the core region, this is not a problem.
xTransLeft = 0.1016  # Inlet x-coordinate in [meters]
xTransRight = 0.1778  # Core region x-coordinate in [meters]
dx_trans = 0.0001524  # Grid size in x direction in [meters]
xCoreLength = 0.0508  # Length of the core region in [meters]
dx_core = 0.0000254 # Grid size in x direction in [meters]
xBufferNum = 120 # Number of buffer layers in x direction
xBufferGR = 1.05  # Growth rate in x direction
# y settings, as the turbulent boundary layer requires much finer grid, we draw the y grid in a turbulent way no matter the flow is turbulent or laminar.
yFirstLayer = 2e-6 # First layer y grid size in [meters]
yEqualNearWallNum = 10 # Number of equal y grid points near the wall
yGRBL = 1.03 # Growth rate in y direction in boundary layer
yEqualDy = 0.0000254 # y grid size in the equal region in [meters]
yEqualHight = 0.0018 # Upper limit of the equal region in [meters] 
yBufferNum = 100 # Number of buffer layers in y direction
yBufferGR = 1.05 # Growth rate in y direction in buffer region
# z settings
zLeft = -0.0022225 # Left boundary of the core region in [meters]
zRight = 0.0022225 # Right boundary of the core region in [meters]
dz = 0.0000175 # Grid size in z direction in [meters]
FILENAME = "BL.xyz"
# multi block settings
x_split = 4  # Number of blocks to split in x direction
WORK_DIR = "input"

def generate_tbl_mesh(xTransLeft=xTransLeft, xTransRight=xTransRight, dx_trans=dx_trans, xCoreLength=xCoreLength, 
                        dx_core=dx_core, xBufferNum=xBufferNum, xBufferGR=xBufferGR,
                        yFirstLayer=yFirstLayer, yEqualNearWallNum=yEqualNearWallNum,
                        yGRBL=yGRBL, yEqualDy=yEqualDy, yEqualHight=yEqualHight,
                        yBufferNum=yBufferNum, yBufferGR=yBufferGR, zLeft=zLeft, zRight=zRight, 
                        dz=dz):
    # Generate full x coordinates first
    xTrans = np.linspace(xTransLeft, xTransRight, int((xTransRight-xTransLeft)/dx_trans) + 1)
    # the core region dx is increased
    xCore = np.linspace(xTransRight, xTransRight + xCoreLength, int(xCoreLength/dx_core) + 1)
    # the buffer region dx is increased by xBufferGR
    xBuffer = [xCore[-1]]
    current_dx = dx_core
    current_x = xCore[-1]
    for i in range(xBufferNum):
        current_dx *= xBufferGR
        xBuffer.append(xBuffer[-1] + current_dx)
    xBuffer = np.array(xBuffer[1:])
    # Combine main domain and buffer zone
    x_full = np.concatenate((xTrans, xCore, xBuffer))
    nx = len(x_full)
    print("nx = ", nx)
    # print x coordinates to file x.txt
    np.savetxt("x.txt", x_full)
    # Calculate points per block (try to make them as equal as possible)
    points_per_block = (nx + x_split - 1) // x_split
    remainder = (nx + x_split - 1) % x_split
    block_sizes = [points_per_block + (1 if i < remainder else 0) for i in range(x_split)]

    # Generate y coordinates
    y = [0]
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
    while current_y < yEqualHight:
        current_y += yEqualDy
        y.append(current_y)
    # Buffer region
    current_y = y[-1]
    current_dy = yEqualDy
    for i in range(yBufferNum):
        current_dy *= yBufferGR
        current_y += current_dy
        y.append(current_y)
    y = np.array(y)
    print("ny = ", len(y))
    # print y coordinates to file y.txt
    np.savetxt("y.txt", y)

    # Generate z coordinates
    z = np.linspace(zLeft, zRight, int((zRight-zLeft)/dz) + 1)
    nz = len(z)
    print("nz = ", nz)
    # print z coordinates to file z.txt
    np.savetxt("z.txt", z)
    
    # Create blocks
    blocks = []
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        x_block = x_full[start_idx:end_idx]
        X, Y, Z = np.meshgrid(x_block, y, z, indexing='ij')
        blocks.append((X, Y, Z))
        start_idx = end_idx - 1  # Overlap one point for block interfaces
        # np.savetxt(f"x_block_{start_idx}.txt", x_block)
    
    print(f"Created {x_split} blocks with sizes: {[b[0].shape[0] for b in blocks]}")
    return blocks

def write_plot3d(filename, blocks):
    # the files should be output to a folder "readGrid-BL"
    # First, create the directory if it doesn't exist
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    with open(os.path.join(WORK_DIR, filename), 'wb') as f:
        # Write number of blocks
        np.array([len(blocks)], dtype=np.int32).tofile(f)
        dimensions = np.array([X.shape for X, _, _ in blocks], dtype=np.int32)
        dimensions.tofile(f)
        
        # Write coordinates for all blocks
        for X, Y, Z in blocks:
            # dims = np.array(X.shape, dtype=np.int32)
            # dims.tofile(f)
            X.ravel(order='F').tofile(f)
            Y.ravel(order='F').tofile(f)
            Z.ravel(order='F').tofile(f)

def write_boundary_conditions(filename, blocks):
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

def generate_laminar_bl_mesh(Lx, nBuffer, dx=dx_core, yFirstLayer=yFirstLayer, yEqualNearWallNum=yEqualNearWallNum,
                             yGRBL=yGRBL, yEqualDy=yEqualDy, yEqualHight=yEqualHight,
                             yBufferNum=yBufferNum, yBufferGR=yBufferGR):
    # Generate y coordinates
    y = [0]
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
    while current_y < yEqualHight:
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
    ny = len(y)
    print("ny = ", ny)

    # Generate x coordinates
    nx = int(Lx / dx) + 1
    x = np.linspace(0, Lx, nx)
    xBuffer = [x[-1]]
    current_dx = dx
    for i in range(nBuffer):
        current_dx *= 1.05
        xBuffer.append(xBuffer[-1] + current_dx)
    xBuffer = np.array(xBuffer[1:])
    # Combine main domain and buffer zone
    x = np.concatenate((x, xBuffer))
    nx = len(x)
    print("nx = ", nx)
    # print x to file x_bl.txt
    np.savetxt(os.path.join(WORK_DIR, "x_bl.txt"), x)

    # Generate z coordinates
    z=[0]
    nz = 1

    # combine x,y,z to 2D mesh
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # X, Y = np.meshgrid(x, y, indexing='ij')
    # print in plot3d format
    with open(os.path.join(WORK_DIR, "bl_2D.xyz"), 'wb') as f:
        # Write number of blocks
        np.array([1], dtype=np.int32).tofile(f)
        # Write dimensions
        dims = np.array([X.shape[0], X.shape[1], 1], dtype=np.int32)
        dims.tofile(f)
        # Write coordinates
        X.ravel(order='F').tofile(f)
        Y.ravel(order='F').tofile(f)
        Z.ravel(order='F').tofile(f)

    # Write boundary conditions
    with open(os.path.join(WORK_DIR, "bl_2D.inp"), 'w') as f:
        f.write(f"1\n")
        # Write number of blocks
        f.write(f"1\n")
        # Write dimensions
        f.write(f"{X.shape[0]}\t{X.shape[1]}\t1\nblock-0\n4\n")
        # Write boundary conditions
        f.write(f"1\t1\t1\t{ny}\t5\n")  # inlet
        f.write(f"{nx}\t{nx}\t1\t{ny}\t6\n") # outlet
        f.write(f"1\t{nx}\t1\t1\t2\n")
        f.write(f"1\t{nx}\t{ny}\t{ny}\t6\n")


def main():
    # Generate the multiblock mesh
    blocks = generate_tbl_mesh()
    
    # Write to Plot3D format
    write_plot3d(FILENAME, blocks)
    
    # # Write boundary conditions
    bc_filename = FILENAME.replace('.xyz', '.inp')
    write_boundary_conditions(bc_filename, blocks)

    # Generate the boundary layer mesh used to compute the inflow condition
    generate_laminar_bl_mesh(0.1778, 100, dx=0.00004)

    GridgenOrPointwise = 0  # 0 for Gridgen, 1 for Pointwise
    dimension = 3  # 2 for 2D, 3 for 3D
    working_directory = WORK_DIR
    gridFile = os.path.join(WORK_DIR, FILENAME)
    boundaryFile = os.path.join(WORK_DIR, FILENAME.replace(".xyz", ".inp"))
    # gridFile = FILENAME
    # boundaryFile = FILENAME.replace('.xyz', '.inp')
    n_proc = x_split
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