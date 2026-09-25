#!/usr/bin/env python3
import sys

from cli import main

from common import common
from config import RunConfig
from modes import Modes, Collector
from driver import solve, solve_both, run
from inputs import (read_layers, read_model, read_specs, specs_source,
                    specs_t0, apply_specs)
from rec_geometry import specs_geometry, specs_azimuth, bjdaz2, glat
from source import double_couple, explosion, packed
from seismogram import seismograms, trace
from greens import (green_tensor, green_tensor_hankel, _bin_index, _propagator,
                    _propagator_hankel)
from modefiles import (write_dispersion, write_eigenfunctions, parse_disp,
                       format_disp, parse_eigen, format_eigen,
                       write_modefile, read_modefile, format_modefile)
from datafiles import (write_grid_ascii, write_grid_bin, read_grid_bin,
                       write_green_ascii, write_green_bin, read_green_bin)

__all__ = [
    "common", "RunConfig", "Modes", "Collector",
    "solve", "solve_both", "run",
    "read_layers", "read_model", "read_specs", "specs_source", "specs_t0",
    "apply_specs", "specs_geometry", "specs_azimuth", "bjdaz2", "glat",
    "double_couple", "explosion", "packed",
    "seismograms", "trace", "green_tensor", "green_tensor_hankel",
    "write_dispersion", "write_eigenfunctions", "parse_disp", "format_disp",
    "parse_eigen", "format_eigen", "write_modefile", "read_modefile",
    "format_modefile",
    "write_grid_ascii", "write_grid_bin", "read_grid_bin",
    "write_green_ascii", "write_green_bin", "read_green_bin",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
