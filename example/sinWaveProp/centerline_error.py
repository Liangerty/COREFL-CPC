from pathlib import Path
import sys
from typing import Callable, Dict, Union

import numpy as np

# Ensure TecplotUtilsGXL is importable when running from the repo root
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from TecplotUtilsGXL import read_tecplot_plt
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "TecplotUtilsGXL.py must be located in the same directory as centerline_error.py"
    ) from exc


def extract_centerline_errors(
    plt_path: Union[str, Path],
    variable_name: str,
    rho_exact_fn: Callable[[float], float],
    x_variable_name: str = "x",
    zone_index: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Extract the centerline of a 2D Tecplot field and compute error norms against an exact solution.

    The Tecplot file is read via TecplotUtilsGXL.read_tecplot_plt. A 2D field with shape
    (NX, NY=7) is expected; the centerline is taken along the middle j-index (NY // 2).

    Args:
        plt_path: Path to the Tecplot .plt file.
        variable_name: Name of the field variable to compare (e.g., "rho").
        rho_exact_fn: Callable giving the exact value rho(x) for a coordinate x.
        x_variable_name: Name of the x-coordinate variable in the Tecplot file.
        zone_index: Zone index to read from the Tecplot dataset.

    Returns:
        Dictionary with x-coordinates, numerical values, exact values, pointwise difference,
        and the L1, L2, and L_inf norms.
    """
    dataset = read_tecplot_plt(str(plt_path), variables=[x_variable_name, variable_name])

    if zone_index < 0 or zone_index >= len(dataset["zones"]):
        raise IndexError(f"Zone index {zone_index} out of range for file {plt_path}")

    zone = dataset["zones"][zone_index]
    I, J, K = zone["dimensions"]

    if J < 1 or K < 1:
        raise ValueError("Zone must have at least one point in each dimension")

    center_j = J // 2  # middle line in the transverse direction

    # Map variable names to their indices in the returned data array
    name_to_idx = {name: idx for idx, name in enumerate(dataset["variables"])}
    if variable_name not in name_to_idx or x_variable_name not in name_to_idx:
        missing = {
            variable_name: variable_name in name_to_idx,
            x_variable_name: x_variable_name in name_to_idx,
        }
        raise ValueError(f"Missing variables in file {plt_path}: {missing}")

    v_idx = name_to_idx[variable_name]
    x_idx = name_to_idx[x_variable_name]

    data = zone["data"]
    rho_numerical = data[v_idx, :, center_j, 0]
    x_values = data[x_idx, :, center_j, 0]

    rho_exact = np.asarray([rho_exact_fn(float(x)) for x in x_values], dtype=float)
    diff = rho_numerical - rho_exact

    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff ** 2)))
    linf = float(np.max(np.abs(diff)))

    return {
        "x": x_values,
        "numerical": rho_numerical,
        "exact": rho_exact,
        "diff": diff,
        "l1": l1,
        "l2": l2,
        "linf": linf,
    }


if __name__ == "__main__":
    # Replace paths and exact function as needed.
    def exact_rho(x_val: float) -> float:
        return 1.5 + np.sin(2.0*3.141592653589793*x_val)

    path_to_plt = "./output/time_series/flowfield_1.0000e+00s.plt"
    try:
        results = extract_centerline_errors(path_to_plt, "density", exact_rho, x_variable_name="x")
        print(f"L_inf: {results['linf']}, L1: {results['l1']}, L2: {results['l2']}")
    except FileNotFoundError:
        print(f"Example file not found: {path_to_plt}")
