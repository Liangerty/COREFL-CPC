# modify this file to make it a importable module.
#
# Add the following lines to make this file importable as a module
__all__ = [
    "read_null_terminated_string",
    "read_tecplot_plt",
    "convert_tecplot_to_latex",
]


import struct
import numpy as np
import re


def read_null_terminated_string(file):
    """
    Reads a null-terminated string from a binary file.
    The function reads 4 bytes at a time (int representation of characters),
    similar to how the C++ MPI version reads strings.

    Args:
        file (file object): Opened binary file.

    Returns:
        str: The decoded string.
    """
    result = []
    while True:
        int_value = struct.unpack("i", file.read(4))[0]  # Read as an integer (4 bytes)
        char = chr(int_value)  # Convert to character
        if char == "\0":  # Stop at null terminator
            break
        result.append(char)

    return "".join(result)


# Mapping of Tecplot escape characters for Greek letters to LaTeX
GREEK_MAP = {
    "a": r"\alpha",
    "b": r"\beta",
    "g": r"\gamma",
    "d": r"\delta",
    "e": r"\epsilon",
    "z": r"\zeta",
    "h": r"\eta",
    "q": r"\theta",
    "i": r"\iota",
    "k": r"\kappa",
    "l": r"\lambda",
    "m": r"\mu",
    "n": r"\nu",
    "x": r"\xi",
    "p": r"\pi",
    "r": r"\rho",
    "s": r"\sigma",
    "t": r"\tau",
    "u": r"\upsilon",
    "f": r"\phi",
    "c": r"\chi",
    "y": r"\psi",
    "w": r"\omega",
    "D": r"\Delta",
    "Q": r"\Theta",
    "L": r"\Lambda",
    "X": r"\Xi",
    "P": r"\Pi",
    "S": r"\Sigma",
    "U": r"\Upsilon",
    "F": r"\Phi",
    "Y": r"\Psi",
    "O": r"\Omega",
}


def convert_tecplot_to_latex(var_name):
    """
    Converts a Tecplot variable name with escape characters into a single LaTeX math expression.

    Args:
        var_name (str): The Tecplot variable name with escape sequences.

    Returns:
        str: Converted variable name in LaTeX format enclosed in a single `$...$` block.
    """

    # Convert Greek letters
    def replace_greek(match):
        greek_char = match.group(1)
        latex_greek = f"{GREEK_MAP.get(greek_char, greek_char)}"
        return latex_greek

    # if the </greek> is not followed by a <sub> or <sup> or < or {, add a space
    var_name = re.sub(r"</greek>([^<\{])", r"</greek> \1", var_name)

    var_name = re.sub(r"<greek>(.*?)</greek>", replace_greek, var_name)

    # Convert subscripts and superscripts (ensuring no extra spaces inside)
    var_name = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", var_name)
    var_name = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", var_name)

    # change all things enclosed by { }, but not after _ or ^ to \widetilde{}
    var_name = re.sub(r"(?<![_\^])\{(.*?)\}", r"\\widetilde{\1}", var_name)
    # change all things enclosed by < > to \overline{}
    var_name = re.sub(r"<(.*?)>", r"\\overline{\1}", var_name)
    # change ^{1/2} to \sqrt{}. Eg. $\\widetilde{u''v''}^{1/2}/\\Delta U$ to $\\sqrt{\\widetilde{u''v''}}/\\Delta U$
    var_name = re.sub(r"(.*?)\^\{1/2\}", r"\\sqrt{\1}", var_name)

    # Ensure a space is added after a Greek letter if followed by a normal character
    # var_name = re.sub(r"(\\[a-zA-Z]+)([a-zA-Z0-9])", r"\1 \2", var_name)

    # Wrap the entire name in a single LaTeX math block
    return f"${var_name}$"

def read_tecplot_plt(filename, variables=None):
    """
    Reads a Tecplot .plt file (structured grid or slice) based on the C++ `FileIO.cpp` and `MyString.cpp`.

    Args:
        filename (str): Path to the Tecplot .plt file.
        variables (list, optional): List of variable names to read. If None, all variables are read.

    Returns:
        dict: Contains title, variables, and zone data.
    """
    with open(filename, "rb") as file:
        # Read magic number (8 bytes)
        magic_bytes = file.read(8)
        magic_str = magic_bytes.decode("utf-8", errors="ignore")
        if not magic_str.startswith("#!TDV"):
            raise ValueError("Not a valid Tecplot .plt file!")

        # Read byte order (4-byte int) and file type (4-byte int)
        byte_order = struct.unpack("i", file.read(4))[0]
        file_type = struct.unpack("i", file.read(4))[0]

        # Read dataset title
        title = read_null_terminated_string(file)

        # Read number of variables (4-byte int)
        num_vars = struct.unpack("i", file.read(4))[0]

        # Read variable names
        var_names = [
            read_null_terminated_string(file)
            # convert_tecplot_to_latex(read_null_terminated_string(file))
            for _ in range(num_vars)
        ]

        # filter variables if specified
        total_variable_number = num_vars
        if variables is not None:
            var_indices = [var_names.index(var) for var in variables if var in var_names]
            num_vars = len(var_indices)
            var_names = [var_names[i] for i in var_indices]
        print(f"Reading {num_vars} variables: {var_names}")

        # Read first marker (4-byte float)
        marker = struct.unpack("f", file.read(4))[0]
        EOH_MARKER = 357.0

        zones = []
        while abs(marker - EOH_MARKER) > 1e-6:
            zone = {}

            # Read zone name
            zone["name"] = read_null_terminated_string(file)
            # zone["name"] = convert_tecplot_to_latex(read_null_terminated_string(file))

            # Skip Parent zones + strand ID (8 bytes)
            file.seek(8, 1)

            # Read solution time (8-byte double)
            zone["solution_time"] = struct.unpack("d", file.read(8))[0]

            # Skip unused variables (20 bytes)
            file.seek(20, 1)

            # Read structured grid dimensions: Imax, Jmax, Kmax (4-byte ints)
            I, J, K = struct.unpack("iii", file.read(12))
            zone["dimensions"] = (I, J, K)

            # Skip one unused int (4 bytes)
            file.seek(4, 1)

            zones.append(zone)

            # Read the next marker (4 bytes)
            marker = struct.unpack("f", file.read(4))[0]

        # Read zone data
        for zone in zones:
            print(f"Reading zone: {zone['name']} with dimensions {zone['dimensions']}")
            marker = struct.unpack("f", file.read(4))[0]
            file.seek(4 * total_variable_number, 1)  # Skip variable data format
            file.seek(12, 1)  # Skip passive_var, var_sharing, connectivity sharing
            # read the variable min/max
            zone["variable_min"] = np.zeros(num_vars)
            zone["variable_max"] = np.zeros(num_vars)
            idx = 0
            for i in range(total_variable_number):
                if i not in var_indices:
                    file.seek(16, 1)  # Skip min/max values
                else:
                    zone["variable_min"][idx] = struct.unpack("d", file.read(8))[0]
                    zone["variable_max"][idx] = struct.unpack("d", file.read(8))[0]
                    idx += 1
            # file.seek(8 * total_variable_number * 2, 1)  # Skip variable max/min

            I, J, K = zone["dimensions"]
            num_points = I * J * K
            # total_values = num_points * num_vars

            # Read data as float32 or float64
            if variables is not None:
              idx = 0
              for l in range(total_variable_number):
                  if l not in var_indices:
                      file.seek(num_points * 8, 1)
                  else:
                      raw_data = file.read(num_points * 8)
                      dtype = np.float64
                      data = np.frombuffer(raw_data, dtype=dtype)
                      if idx == 0:
                          zone["data"] = np.zeros((num_vars, I, J, K), dtype=dtype)
                      # zone["data"][idx] = data.reshape((K, J, I), order="F").transpose(2, 1, 0)
                      zone["data"][idx] = data.reshape((I, J, K), order="F")
                      # zone["data"][idx] = np.array(zone["data"][idx], dtype=dtype)
                      idx += 1
                      # print(f"Reading variable {var_names[l]} with shape {zone["data"][idx].shape}")
            else:
                total_values = num_points * num_vars
                raw_data = file.read(total_values * 8)
                data = np.frombuffer(raw_data, dtype=np.float64)
                zone["data"] = data.reshape((num_vars, I, J, K))
                # to numpy array
                # zone["data"] = np.array(zone["data"], dtype=np.float64)


            # raw_data = file.read(total_values * 8)
            # dtype = np.float64

            # data = np.frombuffer(raw_data, dtype=dtype)
            # zone["data"] = data.reshape((num_vars, I, J, K))

    return {
        "title": title,
        "byte_order": byte_order,
        "file_type": file_type,
        "variables": var_names,
        "zones": zones,
    }

def add_to_zone(zone, data):
    zone["data"] = np.concatenate((zone["data"], data[None, ...]), axis=0)
    zone["variable_min"]= np.append(zone["variable_min"], np.min(data))
    zone["variable_max"]= np.append(zone["variable_max"], np.max(data))

def read_tecplot_datasets(filenames, existing_dataset=None):
    """
    Reads multiple Tecplot .plt files and merges data into the same dataset, ensuring that repeated variables are omitted.

    Args:
        filenames (list): List of Tecplot .plt file paths.
        existing_dataset (dict, optional): Existing dataset to merge new zones into.

    Returns:
        dict: Contains title, variables, and merged zone data.
    """
    if existing_dataset is None:
        merged_dataset = {
            "title": None,
            "byte_order": None,
            "file_type": None,
            "variables": [],
            "zones": {},
        }
    else:
        merged_dataset = existing_dataset

    for filename in filenames:
        dataset = read_tecplot_plt(filename)
        var_set = False

        if merged_dataset["title"] is None:
            merged_dataset["title"] = dataset["title"]
            merged_dataset["byte_order"] = dataset["byte_order"]
            merged_dataset["file_type"] = dataset["file_type"]

        # we assume that all zones in a same file have the same variables
        # we first record the index of the variables in the merged_dataset
        # then we compare the variables in the new dataset with the merged_dataset
        # if the variable is not in the merged_dataset, we add it to the merged_dataset
        # then we add the data to the corresponding zone in the merged_dataset
        index_have_read = len(merged_dataset["variables"])
        index_end_new = []
        i_zone = len(merged_dataset["zones"])
        for zone in dataset["zones"]:
            zone_name = zone["name"]

            for z in range(len(merged_dataset["zones"])):
                if zone_name == merged_dataset["zones"][z]["name"]:
                    # Merge data while avoiding duplicate variables
                    existing_vars = merged_dataset["variables"]
                    new_vars = dataset["variables"]
                    unique_vars = []

                    for i, var in enumerate(new_vars):
                        if var not in existing_vars:
                            unique_vars.append(var)
                            # unique_data.append(zone['data'][i])
                            index_end_new += [
                                (i, index_have_read + len(unique_vars) - 1)
                            ]

                    merged_dataset["variables"].extend(unique_vars)
                    merged_dataset["zones"][z]["data"] = np.concatenate(
                        # use the index_end_new to get the correct data
                        (
                            merged_dataset["zones"][z]["data"],
                            np.array([zone["data"][i] for i, _ in index_end_new]),
                        ),
                        axis=0,
                    )
                    break
                    # merged_dataset['zones'][z]['data'] = np.concatenate(
                    #     (merged_dataset['zones'][z]['data'], np.array(unique_data)), axis=0
                    # )
                    # break
                else:
                    continue
            else:
                # if not var_set, set the variables to dataset['variables']
                if not var_set:
                    merged_dataset["variables"] = dataset["variables"]
                    var_set = True
                merged_dataset["zones"][i_zone] = zone
                i_zone += 1

            # if zone_name not in merged_dataset['zones']:
            #     merged_dataset['zones'][zone_name] = {
            #         'dimensions': zone['dimensions'],
            #         'solution_time': zone['solution_time'],
            #         'variables': dataset['variables'],
            #         'data': zone['data']
            #     }
            # else:
            #     # Merge data while avoiding duplicate variables
            #     existing_vars = merged_dataset['zones'][zone_name]['variables']
            #     new_vars = dataset['variables']
            #     unique_vars = []
            #     unique_data = []

            #     for i, var in enumerate(new_vars):
            #         if var not in existing_vars:
            #             unique_vars.append(var)
            #             unique_data.append(zone['data'][i])

            #     merged_dataset['zones'][zone_name]['variables'].extend(unique_vars)
            #     merged_dataset['zones'][zone_name]['data'] = np.concatenate(
            #         (merged_dataset['zones'][zone_name]['data'], np.array(unique_data)), axis=0
            #     )

    # merged_dataset['variables'] = list(set(merged_dataset['variables']))

    return {
        "title": merged_dataset["title"],
        "byte_order": merged_dataset["byte_order"],
        "file_type": merged_dataset["file_type"],
        "variables": merged_dataset["variables"],
        "zones": [
            {
                "name": merged_dataset["zones"][i]["name"],
                "dimensions": merged_dataset["zones"][i]["dimensions"],
                "solution_time": merged_dataset["zones"][i]["solution_time"],
                "data": merged_dataset["zones"][i]["data"],
            }
            for i in range(len(merged_dataset["zones"]))
        ],
    }


def write_null_terminated_string(file, string):
    """
    Writes a null-terminated string to a binary file, 4 bytes at a time (int representation of characters),
    similar to how the C++ MPI version writes strings.

    Args:
        file (file object): Opened binary file for writing.
        string (str): The string to write.
    """
    for char in string:
        int_value = ord(char)
        file.write(struct.pack("i", int_value))
    file.write(struct.pack("i", 0))  # Null terminator


def write_tecplot_plt(filename, data, variables=None, title = None):
    """
    Writes a Tecplot .plt file (structured grid or slice) in the same format as read_tecplot_plt reads.

    Args:
        filename (str): Path to the output Tecplot .plt file.
        data (dict): Dictionary with keys 'title', 'byte_order', 'file_type', 'variables', 'zones'.
        variables (list, optional): List of variable names to write. If None, all variables are written.
    """
    # Determine which variables to write
    all_var_names = data["variables"]
    if variables is not None:
        var_indices = [all_var_names.index(var) for var in variables if var in all_var_names]
        out_var_names = [all_var_names[i] for i in var_indices]
    else:
        var_indices = list(range(len(all_var_names)))
        out_var_names = all_var_names

    with open(filename, "wb") as file:
        # Write magic number (8 bytes)
        file.write(b"#!TDV112")

        # Write byte order (4-byte int) and file type (4-byte int)
        file.write(struct.pack("i", 1))
        file.write(struct.pack("i", 0))

        # Write dataset title
        write_null_terminated_string(file, title if title is not None else "field_data")

        # Write number of variables (4-byte int)
        num_vars = len(out_var_names)
        file.write(struct.pack("i", num_vars))

        # Write variable names
        for var in out_var_names:
            write_null_terminated_string(file, var)

        # Write zones
        for zone in data["zones"]:
            # Write first marker (4-byte float)
            file.write(struct.pack("f", 299.0))  # Zone marker (Tecplot convention)

            # Write zone name
            write_null_terminated_string(file, zone["name"])

            # Write Parent zones + strand ID (8 bytes, set to 0)
            file.write(struct.pack("i", -1))
            file.write(struct.pack("i", -2))

            # Write solution time (8-byte double)
            file.write(struct.pack("d", zone["solution_time"]))

            # Write unused variables (20 bytes, set to 0)
            file.write(struct.pack("i", -1))
            file.write(struct.pack("i", 0))
            file.write(struct.pack("i", 0))
            file.write(struct.pack("i", 0))
            file.write(struct.pack("i", 0))

            # Write structured grid dimensions: Imax, Jmax, Kmax (4-byte ints)
            I, J, K = zone["dimensions"]
            file.write(struct.pack("iii", I, J, K))

            # Write one unused int (4 bytes, set to 0)
            file.write(struct.pack("i", 0))

        # Write end of header marker (4 bytes, EOH_MARKER)
        file.write(struct.pack("f", 357.0))

        # Write zone data
        for zone in data["zones"]:
            # Write zone data marker (4 bytes, 299.0)
            file.write(struct.pack("f", 299.0))
            I, J, K = zone["dimensions"]
            num_points = I * J * K

            # Write variable data format (4 bytes per variable, set to 2 for double/float64)
            for _ in range(num_vars):
                file.write(struct.pack("i", 2))

            # Write passive_var, var_sharing, connectivity sharing (12 bytes, set to 0)
            file.write(struct.pack("i", 0))
            file.write(struct.pack("i", 0))
            file.write(struct.pack("i", -1))

            # Write variable min/max
            for i in range(num_vars):
                file.write(struct.pack("d", zone["variable_min"][i]))
                file.write(struct.pack("d", zone["variable_max"][i]))

            # Write data as float64, Fortran order
            data_arr = zone["data"]
            # Ensure shape is (all_vars, I, J, K)
            assert data_arr.shape[1:] == (I, J, K), f"Data shape mismatch: {data_arr.shape} vs ({len(all_var_names)}, {I}, {J}, {K})"
            # Write only selected variables
            for idx in var_indices:
                file.write(data_arr[idx].astype(np.float64).flatten(order="F"))
