"""
Script for sorting out files for the data release
"""
from pathlib import Path
from shutil import copy
from sys import argv

source = Path(argv[1])
datafiles = sorted(source.glob("*.Raw"))
identifier = f"{source.parts[-2]}/{source.parts[-1]}"

nrfiles = len(datafiles)
divider = nrfiles // 5
selectedfiles = datafiles[::divider][:5]

destination_general = Path("/media/oli4/O4B/Zenodo/BlackFly")
destination_here = destination_general / identifier
destination_here.mkdir(parents=True, exist_ok=True)

for file in selectedfiles:
    newfile = destination_here / file.name
    copy(file, newfile)
    print("Copied to", newfile)
