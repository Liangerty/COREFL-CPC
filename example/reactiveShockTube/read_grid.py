"""
Python reimplementation of the C++ main in main.cpp.
Provides a callable function to generate boundary_condition and grid outputs
given explicit parameters (no parameter file required), while preserving the
original logic and formatting.
"""
from __future__ import annotations

import os
import struct
from array import array
from typing import Iterable, Iterator, List


class TokenStream:
    def __init__(self, iterable: Iterable[str]):
        self._iter = iter(iterable)

    def __next__(self) -> str:
        return next(self._iter)

    def next_int(self) -> int:
        return int(next(self))

    def next_float(self) -> float:
        return float(next(self))


def token_generator(file_obj: Iterable[str]) -> Iterator[str]:
    for line in file_obj:
        for tok in line.split():
            yield tok


def read_ascii_grid(
    grid_file: str, gridgen_or_pointwise: int, set_z: bool, z_value: float
):
    with open(grid_file, "r", encoding="utf-8") as f:
        tokens = TokenStream(token_generator(f))
        n_block = tokens.next_int()
        ni: List[int] = [0] * n_block
        nj: List[int] = [0] * n_block
        nk: List[int] = [0] * n_block
        n_grid: List[int] = [0] * n_block
        order: List[int] = [-1] * n_block
        x: List[array] = [array("d") for _ in range(n_block)]
        y: List[array] = [array("d") for _ in range(n_block)]
        z: List[array] = [array("d") for _ in range(n_block)]

        if gridgen_or_pointwise == 0:
            for b in range(n_block):
                ni[b] = tokens.next_int()
                nj[b] = tokens.next_int()
                nk[b] = tokens.next_int()
                n_grid[b] = ni[b] * nj[b] * nk[b]

        for b in range(n_block):
            if gridgen_or_pointwise == 1:
                ni[b] = tokens.next_int()
                nj[b] = tokens.next_int()
                nk[b] = tokens.next_int()
                n_grid[b] = ni[b] * nj[b] * nk[b]

            for i in range(b):
                if n_grid[b] > n_grid[order[i]]:
                    for j in range(b, i, -1):
                        order[j] = order[j - 1]
                    order[i] = b
                    break
            if order[b] == -1:
                order[b] = b

            count = n_grid[b]
            x[b].extend(tokens.next_float() for _ in range(count))
            y[b].extend(tokens.next_float() for _ in range(count))
            if set_z:
                if abs(z_value) < 1e-10:
                    z[b].extend(0.0 for _ in range(count))
                else:
                    z[b].extend(z_value for _ in range(count))
                for _ in range(count):
                    next(tokens)  # consume original z values
            else:
                z[b].extend(tokens.next_float() for _ in range(count))
    return n_block, ni, nj, nk, x, y, z, n_grid, order


def read_binary_grid(
    grid_file: str, gridgen_or_pointwise: int, set_z: bool, z_value: float
):
    with open(grid_file, "rb") as f:
        n_block_bytes = f.read(4)
        n_block = struct.unpack("i", n_block_bytes)[0]
        ni: List[int] = [0] * n_block
        nj: List[int] = [0] * n_block
        nk: List[int] = [0] * n_block
        n_grid: List[int] = [0] * n_block
        order: List[int] = [-1] * n_block
        x: List[array] = [array("d") for _ in range(n_block)]
        y: List[array] = [array("d") for _ in range(n_block)]
        z: List[array] = [array("d") for _ in range(n_block)]

        if gridgen_or_pointwise == 0:
            for b in range(n_block):
                ni[b], nj[b], nk[b] = struct.unpack("iii", f.read(12))
                n_grid[b] = ni[b] * nj[b] * nk[b]

        for b in range(n_block):
            if gridgen_or_pointwise == 1:
                ni[b], nj[b], nk[b] = struct.unpack("iii", f.read(12))
                n_grid[b] = ni[b] * nj[b] * nk[b]

            for i in range(b):
                if n_grid[b] > n_grid[order[i]]:
                    for j in range(b, i, -1):
                        order[j] = order[j - 1]
                    order[i] = b
                    break
            if order[b] == -1:
                order[b] = b

            count = n_grid[b]
            x[b].fromfile(f, count)
            y[b].fromfile(f, count)
            if set_z:
                if abs(z_value) < 1e-10:
                    z[b].extend(0.0 for _ in range(count))
                    f.seek(8 * count, os.SEEK_CUR)
                else:
                    z[b].extend(z_value for _ in range(count))
                    f.seek(8 * count, os.SEEK_CUR)
            else:
                z[b].fromfile(f, count)
    return n_block, ni, nj, nk, x, y, z, n_grid, order


def read_boundary_file(boundary_file: str, n_block: int, dimension: int):
    def read_line_values(handle, expected: int) -> List[int]:
        while True:
            line = handle.readline()
            if line == "":
                raise EOFError("Unexpected end of boundary file")
            parts = line.strip().split()
            if parts:
                return [int(val) for val in parts[:expected]]

    n_inter = [0 for _ in range(n_block)]
    inter_id: List[List[int]] = [[] for _ in range(n_block)]
    range_boundary: List[List[List[int]]] = [[] for _ in range(n_block)]
    boundary_type: List[List[int]] = [[] for _ in range(n_block)]
    range_interior: List[List[List[int]]] = [[] for _ in range(n_block)]
    range_interior_target: List[List[List[int]]] = [[] for _ in range(n_block)]
    interior_target_id: List[List[int]] = [[] for _ in range(n_block)]

    with open(boundary_file, "r", encoding="utf-8") as f:
        _ = f.readline()
        _ = f.readline()
        for b in range(n_block):
            f.readline()
            f.readline()
            # f.readline()
            n_bound_line = f.readline()
            n_bound = int(n_bound_line.strip().split()[0])

            for _ in range(n_bound):
                vals = read_line_values(f, 5 if dimension == 2 else 7)
                x1, x2, y1, y2 = vals[0], vals[1], vals[2], vals[3]
                if dimension == 2:
                    z1 = z2 = 1
                    i_type = vals[4]
                else:
                    z1, z2, i_type = vals[4], vals[5], vals[6]

                if i_type < 0:
                    target_vals = read_line_values(f, 5 if dimension == 2 else 7)
                    tx1, tx2, ty1, ty2 = (
                        target_vals[0],
                        target_vals[1],
                        target_vals[2],
                        target_vals[3],
                    )
                    if dimension == 2:
                        tz1 = tz2 = 1
                        t_type = target_vals[4]
                    else:
                        tz1, tz2, t_type = target_vals[4], target_vals[5], target_vals[6]

                    range_interior[b].append([x1, x2, y1, y2, z1, z2])
                    range_interior_target[b].append([tx1, tx2, ty1, ty2, tz1, tz2])
                    interior_target_id[b].append(t_type - 1)
                    n_inter[b] += 1
                    inter_id[b].append(t_type)
                    # Sync pairing info if target block already processed
                    target_block_idx = t_type - 1
                    this_face = n_inter[b] - 1
                    if target_block_idx < b:
                        found_face = -1
                        for tf in range(n_inter[target_block_idx]):
                            il1 = abs(range_interior[target_block_idx][tf][0])
                            ir1 = abs(range_interior[target_block_idx][tf][1])
                            jl1 = abs(range_interior[target_block_idx][tf][2])
                            jr1 = abs(range_interior[target_block_idx][tf][3])
                            kl1 = abs(range_interior[target_block_idx][tf][4])
                            kr1 = abs(range_interior[target_block_idx][tf][5])

                            il2 = abs(range_interior_target[b][this_face][0])
                            ir2 = abs(range_interior_target[b][this_face][1])
                            jl2 = abs(range_interior_target[b][this_face][2])
                            jr2 = abs(range_interior_target[b][this_face][3])
                            kl2 = abs(range_interior_target[b][this_face][4])
                            kr2 = abs(range_interior_target[b][this_face][5])

                            il_match = (il1 == il2 and ir1 == ir2) or (il1 == ir2 and ir1 == il2)
                            jl_match = (jl1 == jl2 and jr1 == jr2) or (jl1 == jr2 and jr1 == jl2)
                            if il_match and jl_match:
                                if dimension == 2:
                                    found_face = tf
                                    break
                                kl_match = (kl1 == kl2 and kr1 == kr2) or (kl1 == kr2 and kr1 == kl2)
                                if kl_match:
                                    found_face = tf
                                    break
                        if found_face != -1:
                            range_interior_target[b][this_face] = range_interior[target_block_idx][found_face]
                            range_interior[b][this_face] = range_interior_target[target_block_idx][found_face]
                else:
                    if dimension == 2:
                        range_boundary[b].append([x1 - 1, x2 - 1, y1 - 1, y2 - 1, 0, 0])
                    else:
                        range_boundary[b].append([x1 - 1, x2 - 1, y1 - 1, y2 - 1, z1 - 1, z2 - 1])
                    boundary_type[b].append(i_type)
    return (
        n_inter,
        inter_id,
        range_boundary,
        boundary_type,
        range_interior,
        range_interior_target,
        interior_target_id,
    )


def allocate_blocks(n_proc: int, n_block: int, order: List[int], n_grid: List[int]):
    n_block_proc = [0 for _ in range(n_proc)]
    block_grid_num = [0 for _ in range(n_proc)]
    block_id: List[List[int]] = [[] for _ in range(n_proc)]
    block_in_which_proc = [0 for _ in range(n_block)]
    block_label_in_proc = [0 for _ in range(n_block)]

    for p in range(n_proc):
        block_id[p].append(order[p])
        n_block_proc[p] = 1
        block_grid_num[p] = n_grid[order[p]]
        block_in_which_proc[order[p]] = p
        block_label_in_proc[order[p]] = 0

    if n_proc < n_block:
        for b in range(n_proc, n_block):
            p = min(range(n_proc), key=lambda idx: block_grid_num[idx])
            block_id[p].append(order[b])
            n_block_proc[p] += 1
            block_grid_num[p] += n_grid[order[b]]
            block_in_which_proc[order[b]] = p
            block_label_in_proc[order[b]] = n_block_proc[p] - 1

    for p in range(n_proc):
        block_id[p].sort()
        for label, block in enumerate(block_id[p]):
            block_label_in_proc[block] = label
    return (
        n_block_proc,
        block_grid_num,
        block_id,
        block_in_which_proc,
        block_label_in_proc,
    )


def write_boundary_files(
    n_proc: int,
    n_block_proc: List[int],
    block_id: List[List[int]],
    range_boundary: List[List[List[int]]],
    boundary_type: List[List[int]],
    output_dir: str,
):
    boundary_dir = os.path.join(output_dir, "boundary_condition")
    os.makedirs(boundary_dir, exist_ok=True)
    for p in range(n_proc):
        file_name = os.path.join(boundary_dir, f"boundary{p:4d}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            for b in range(n_block_proc[p]):
                block = block_id[p][b]
                n_boundary = len(boundary_type[block])
                f.write(f"{n_boundary:7d}\n")
                for idx in range(n_boundary):
                    vals = range_boundary[block][idx]
                    f.write(
                        f"{vals[0]:7d}{vals[1]:7d}{vals[2]:7d}{vals[3]:7d}{vals[4]:7d}{vals[5]:7d}{boundary_type[block][idx]:7d}\n"
                    )


def write_grid_ascii(
    n_proc: int,
    n_block_proc: List[int],
    block_id: List[List[int]],
    ni: List[int],
    nj: List[int],
    nk: List[int],
    x: List[array],
    y: List[array],
    z: List[array],
    output_dir: str,
):
    grid_dir = os.path.join(output_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)

    def write_wrapped_values(file_obj, values: array, start_count: int = 0, per_line: int = 3) -> int:
        count = start_count
        for val in values:
            file_obj.write(f"{val:e}\t")
            count += 1
            if count % per_line == 0:
                file_obj.write("\n")
        return count

    for p in range(n_proc):
        file_name = os.path.join(grid_dir, f"grid{p:4d}.grd")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"{n_block_proc[p]}\n")
            for block in block_id[p]:
                f.write(f"{ni[block]}\t{nj[block]}\t{nk[block]}\n")
            for block in block_id[p]:
                line_count = 0
                line_count = write_wrapped_values(f, x[block], line_count)
                line_count = write_wrapped_values(f, y[block], line_count)
                line_count = write_wrapped_values(f, z[block], line_count)


def write_grid_binary(
    n_proc: int,
    n_block_proc: List[int],
    block_id: List[List[int]],
    ni: List[int],
    nj: List[int],
    nk: List[int],
    x: List[array],
    y: List[array],
    z: List[array],
    output_dir: str,
):
    grid_dir = os.path.join(output_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)
    for p in range(n_proc):
        file_name = os.path.join(grid_dir, f"grid{p:4d}.dat")
        with open(file_name, "wb") as f:
            f.write(struct.pack("i", n_block_proc[p]))
            for block in block_id[p]:
                f.write(struct.pack("iii", ni[block], nj[block], nk[block]))
            for block in block_id[p]:
                f.write(x[block].tobytes())
                f.write(y[block].tobytes())
                f.write(z[block].tobytes())


def compute_outer_inner(
    n_block: int,
    n_inter: List[int],
    interior_target_id: List[List[int]],
    range_interior: List[List[List[int]]],
    range_interior_target: List[List[List[int]]],
    block_in_which_proc: List[int],
):
    n_outer = [0 for _ in range(n_block)]
    n_inner = [0 for _ in range(n_block)]
    communicate_label: List[List[int]] = [[] for _ in range(n_block)]
    outer_face_label: List[List[int]] = [[] for _ in range(n_block)]
    inner_face_label: List[List[int]] = [[] for _ in range(n_block)]
    outer_counter = 0

    for b in range(n_block):
        n_face = n_inter[b]
        for f_idx in range(n_face):
            if block_in_which_proc[interior_target_id[b][f_idx]] == block_in_which_proc[b]:
                inner_face_label[b].append(f_idx)
                n_inner[b] += 1
            else:
                outer_face_label[b].append(f_idx)
                n_outer[b] += 1
                communicate_label[b].append(outer_counter)
                outer_counter += 1
    return n_outer, n_inner, communicate_label, outer_face_label, inner_face_label


def find_target_face(
    b: int,
    f_idx: int,
    interior_target_id: List[List[int]],
    range_interior: List[List[List[int]]],
    range_interior_target: List[List[List[int]]],
    outer_face_label: List[List[int]],
    n_outer: List[int],
):
    target_block = interior_target_id[b][f_idx]
    target_face = -1
    target_range = range_interior_target[b][f_idx]
    for i in range(n_outer[target_block]):
        f_tar = outer_face_label[target_block][i]
        if interior_target_id[target_block][f_tar] != b:
            continue
        rng = range_interior[target_block][f_tar]
        if rng[0] == target_range[0] and rng[1] == target_range[1] and rng[2] == target_range[2] and rng[3] == target_range[3] and rng[4] == target_range[4] and rng[5] == target_range[5]:
            target_face = i
            break
        elif rng[0] == target_range[1] and rng[1] == target_range[0] and rng[2] == target_range[2] and rng[3] == target_range[3] and rng[4] == target_range[4] and rng[5] == target_range[5]:
            rng[0], rng[1] = rng[1], rng[0]
            target_face = i
            break
        elif rng[0] == target_range[0] and rng[1] == target_range[1] and rng[2] == target_range[3] and rng[3] == target_range[2] and rng[4] == target_range[4] and rng[5] == target_range[5]:
            rng[2], rng[3] = rng[3], rng[2]
            target_face = i
            break
        elif rng[0] == target_range[0] and rng[1] == target_range[1] and rng[2] == target_range[2] and rng[3] == target_range[3] and rng[4] == target_range[5] and rng[5] == target_range[4]:
            rng[4], rng[5] = rng[5], rng[4]
            target_face = i
            break
        elif rng[0] == target_range[1] and rng[1] == target_range[0] and rng[2] == target_range[3] and rng[3] == target_range[2] and rng[4] == target_range[4] and rng[5] == target_range[5]:
            rng[0], rng[1] = rng[1], rng[0]
            rng[2], rng[3] = rng[3], rng[2]
            target_face = i
            break
        elif rng[0] == target_range[1] and rng[1] == target_range[0] and rng[2] == target_range[2] and rng[3] == target_range[3] and rng[4] == target_range[5] and rng[5] == target_range[4]:
            rng[0], rng[1] = rng[1], rng[0]
            rng[4], rng[5] = rng[5], rng[4]
            target_face = i
            break
        elif rng[0] == target_range[0] and rng[1] == target_range[1] and rng[2] == target_range[3] and rng[3] == target_range[2] and rng[4] == target_range[5] and rng[5] == target_range[4]:
            rng[2], rng[3] = rng[3], rng[2]
            rng[4], rng[5] = rng[5], rng[4]
            target_face = i
            break
    return target_block, target_face


def write_parallel_files(
    n_proc: int,
    n_block_proc: List[int],
    block_id: List[List[int]],
    n_outer: List[int],
    range_interior: List[List[List[int]]],
    range_interior_target: List[List[List[int]]],
    interior_target_id: List[List[int]],
    communicate_label: List[List[int]],
    block_in_which_proc: List[int],
    outer_face_label: List[List[int]],
    output_dir: str,
):
    boundary_dir = os.path.join(output_dir, "boundary_condition")
    os.makedirs(boundary_dir, exist_ok=True)
    for p in range(n_proc):
        n_send = sum(n_outer[block_id[p][b]] for b in range(n_block_proc[p]))
        file_name = os.path.join(boundary_dir, f"parallel{p:4d}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"{n_block_proc[p]:9d}{n_send:9d}\n")
            for b in range(n_block_proc[p]):
                f.write(f"{n_outer[block_id[p][b]]:9d}")
            f.write("\n")
            f.write("iMin  iMax  jMin  jMax  kMin  kMax   s_r_id  send_flag  recv_flag\n")
            for b_in_p in range(n_block_proc[p]):
                b_global = block_id[p][b_in_p]
                for counter in range(n_outer[b_global]):
                    f_idx = outer_face_label[b_global][counter]
                    rng = range_interior[b_global][f_idx]
                    f.write(
                        f"{rng[0]:9d}{rng[1]:9d}{rng[2]:9d}{rng[3]:9d}{rng[4]:9d}{rng[5]:9d}{block_in_which_proc[interior_target_id[b_global][f_idx]]:9d}{communicate_label[b_global][counter]:9d}"
                    )
                    target_block, target_face = find_target_face(
                        b_global,
                        f_idx,
                        interior_target_id,
                        range_interior,
                        range_interior_target,
                        outer_face_label,
                        n_outer,
                    )
                    if target_face == -1:
                        print(f"Error: Cannot find the target face of block {b_global}, face {f_idx}.")
                        f.write(f"{target_face:9d}\n")
                    else:
                        f.write(f"{communicate_label[target_block][target_face]:9d}\n")


def write_inner_files(
    n_proc: int,
    n_block_proc: List[int],
    block_id: List[List[int]],
    n_inner: List[int],
    range_interior: List[List[List[int]]],
    range_interior_target: List[List[List[int]]],
    block_label_in_proc: List[int],
    inner_face_label: List[List[int]],
    output_dir: str,
):
    boundary_dir = os.path.join(output_dir, "boundary_condition")
    os.makedirs(boundary_dir, exist_ok=True)
    for p in range(n_proc):
        n_send = sum(n_inner[block_id[p][b]] for b in range(n_block_proc[p]))
        file_name = os.path.join(boundary_dir, f"inner{p:4d}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"{n_send:9d}\n")
            for b in range(n_block_proc[p]):
                f.write(f"{n_inner[block_id[p][b]]:9d}\t")
            f.write("\n")
            f.write("iMin  iMax  jMin  jMax  kMin  kMax   id\n")
            if n_send > 0:
                for b_in_p in range(n_block_proc[p]):
                    b_global = block_id[p][b_in_p]
                    for idx in range(n_inner[b_global]):
                        f_idx = inner_face_label[b_global][idx]
                        rng = range_interior[b_global][f_idx]
                        f.write(
                            f"{rng[0]:9d}{rng[1]:9d}{rng[2]:9d}{rng[3]:9d}{rng[4]:9d}{rng[5]:9d}{block_label_in_proc[b_global] + 1:9d}\n"
                        )
                        trg = range_interior_target[b_global][f_idx]
                        f.write(
                            f"{trg[0]:9d}{trg[1]:9d}{trg[2]:9d}{trg[3]:9d}{trg[4]:9d}{trg[5]:9d}{block_label_in_proc[interior_target_id[b_global][f_idx]] + 1:9d}\n"
                        )


def write_info_file(
    n_proc: int,
    n_block_proc: List[int],
    block_grid_num: List[int],
    block_id: List[List[int]],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, "grid_allocate_info.txt")
    with open(info_path, "w", encoding="utf-8") as info:
        for p in range(n_proc):
            info.write(
                f"{p} grid file:\n" f"total grid number is {block_grid_num[p]}\n" f"block number is {n_block_proc[p]}\n" f"block id is "
            )
            for block in block_id[p]:
                info.write(f"{block} ")
            info.write("\n")


def read_grid(
    gridgen_or_pointwise: int,
    dimension: int,
    grid_file_name: str,
    boundary_file_name: str,
    n_proc: int,
    is_binary: bool,
    write_binary: bool,
    set_z: bool = False,
    z_value: float = 0.0,
    output_dir: str = "./input",
) -> None:
    """Entry point callable from other modules; mirrors C++ main behavior."""
    if set_z and dimension != 2:
        set_z = False

    if is_binary:
        n_block, ni, nj, nk, x, y, z, n_grid, order = read_binary_grid(
            grid_file_name, gridgen_or_pointwise, set_z and dimension == 2, z_value
        )
    else:
        n_block, ni, nj, nk, x, y, z, n_grid, order = read_ascii_grid(
            grid_file_name, gridgen_or_pointwise, set_z and dimension == 2, z_value
        )

    if n_block < n_proc:
        raise RuntimeError("The number of blocks is smaller than the number of processes.")

    print("Finish reading grid.")
    if set_z and dimension == 2:
        print(f"Set z value to {z_value:e}.")

    (
        n_inter,
        inter_id,
        range_boundary,
        boundary_type,
        range_interior,
        range_interior_target,
        interior_target_id,
    ) = read_boundary_file(boundary_file_name, n_block, dimension)

    (
        n_block_proc,
        block_grid_num,
        block_id,
        block_in_which_proc,
        block_label_in_proc,
    ) = allocate_blocks(n_proc, n_block, order, n_grid)

    write_boundary_files(n_proc, n_block_proc, block_id, range_boundary, boundary_type, output_dir)
    print("Finishing writing boundary condition files.")

    if not write_binary:
        write_grid_ascii(n_proc, n_block_proc, block_id, ni, nj, nk, x, y, z, output_dir)
    else:
        write_grid_binary(n_proc, n_block_proc, block_id, ni, nj, nk, x, y, z, output_dir)

    write_info_file(n_proc, n_block_proc, block_grid_num, block_id, output_dir)
    total_grid = sum(block_grid_num)
    print(f"Total grid number is {total_grid}.")

    n_outer, n_inner, communicate_label, outer_face_label, inner_face_label = compute_outer_inner(
        n_block, n_inter, interior_target_id, range_interior, range_interior_target, block_in_which_proc
    )

    write_parallel_files(
        n_proc,
        n_block_proc,
        block_id,
        n_outer,
        range_interior,
        range_interior_target,
        interior_target_id,
        communicate_label,
        block_in_which_proc,
        outer_face_label,
        output_dir,
    )

    write_inner_files(
        n_proc,
        n_block_proc,
        block_id,
        n_inner,
        range_interior,
        range_interior_target,
        block_label_in_proc,
        inner_face_label,
        output_dir,
    )

    print("Grid file has been saved successfully!!")


# Convenience wrapper for quick manual use; other modules should call generate_grid directly.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python port of readGrid main")
    parser.add_argument("--GridgenOrPointwise", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--gridFile", required=True)
    parser.add_argument("--boundaryFile", required=True)
    parser.add_argument("--n_proc", type=int, required=True)
    parser.add_argument("--isBinary", type=int, choices=[0, 1], required=True)
    parser.add_argument("--writeBinary", type=int, choices=[0, 1], required=True)
    parser.add_argument("--setZ", type=int, choices=[0, 1], default=0)
    parser.add_argument("--zValue", type=float, default=0.0)
    parser.add_argument("--output_dir", default="./input")
    args = parser.parse_args()

    read_grid(
        gridgen_or_pointwise=args.GridgenOrPointwise,
        dimension=args.dimension,
        grid_file_name=args.gridFile,
        boundary_file_name=args.boundaryFile,
        n_proc=args.n_proc,
        is_binary=bool(args.isBinary),
        write_binary=bool(args.writeBinary),
        set_z=bool(args.setZ),
        z_value=args.zValue,
        output_dir=args.output_dir,
    )
