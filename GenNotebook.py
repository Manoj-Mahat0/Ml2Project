# write python script to convert python source file into ipynb
# python source is split by "### CELL\n" delimiter indicating different cells.
# implement this script

# prompt:
# write a python source file to be formatted as jupyter notebook
# by differentiating cells with "### CELL\n" delimiter

import json
import sys
import re

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        content = f.read()

    # Use regex to split the content by the cell delimiter (with or without number)
    cells = re.split(r'### CELL ?\d*\n', content)

    # Create the notebook structure
    notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Create a code cell for each split section
    for cell in cells:
        # Strip whitespace and check if the cell is not empty
        cell_content = cell.strip()
        if cell_content:
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_content.splitlines(keepends=True),
            })

    # Write the notebook to the output file
    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_ipynb.py <source.py> <output.ipynb>")
        sys.exit(1)

    py_file = sys.argv[1]
    ipynb_file = sys.argv[2]

    convert_py_to_ipynb(py_file, ipynb_file)
    print(f"Converted {py_file} to {ipynb_file}.")
