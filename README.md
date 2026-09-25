# SW1D

Surface waves in a 1D layered elastic half-space.

---

## Requirements

| | |
|---|---|
| Python | 3.8 or newer (developed on 3.14) |
| numpy | required by the solver |
| matplotlib | required by the scripts in `plotting/` only |

---

## Structure

`SW1D.py` is the script you run.

| | |
|---|---|
| `common.py` | the model arrays and counters the code shares |
| `model.py` | deals with the layer interfaces at the source and receiver depths, earth-flattening, Q dispersion, velocity bounds |
| `propagator.py` | crossing one layer, and the recursion that counts branch crossings |
| `modecount.py` | how many branches lie below a trial phase velocity |
| `search.py` | bracketing a branch's phase velocity, then refining it |
| `eigenfunctions.py` | rebuilding the eigenfunction at a converged root |
| `dispersion.py` | the dispersion determinant and the eigenfunction pass |
| `integrals.py` | Rayleigh energy integrals and partial derivatives |
| `driver.py` | sweeping the frequency grid, branch by branch |
| `config.py` | run configuration |
| `modes.py` | `Modes`, and the object that collects them |
| `inputs.py` | reading the model file and `event.specs` |
| `rec_geometry.py` | receiver distances and the source-to-receiver azimuth |
| `source.py` | fault angles or a moment tensor |
| `modefiles.py` | the `disp.*` and `eigen.*` files, text and binary |
| `datafiles.py` | the seismogram and Green's tensor files |
| `seismogram.py` | summing modes into displacement spectra |
| `greens.py` | the Green's tensor |
| `cli.py` | argument parsing and the three commands |


---

## Input files

### 1. The model file

```
line 1        n0   iefl   tref
n0 lines      d    vp   vs   rho   Q_beta   Q_alpha
1 line        c1   c2   nbran1   nbran2
1 line        nsrc   nom   df   fo
1 line        source depth(s)          nsrc values
1 line        receiver depth
```

| symbol | meaning | units |
|---|---|---|
| `n0` | number of layers, including the halfspace | |
| `iefl` | `1` applies earth flattening, `0` does not | |
| `tref` | reference period for the physical-dispersion correction that goes with Q (`0` to disable) | s |
| `d` | layer thickness (`0` indicates halfspace) | km |
| `vp`, `vs` | P and S velocity of the layer | km/s |
| `rho` | density | g/cm³ |
| `Q_beta`, `Q_alpha` | quality factors | |
| `c1`, `c2` | lower and upper bound of the phase-velocity search (`0 0` to let the solver pick) | km/s |
| `nbran1`, `nbran2` | lowest and highest mode branch, both inclusive (negative `nbran2` means every branch that exists) | |
| `nsrc` | number of source depths (at most 4) | |
| `nom` | number of frequencies | |
| `df` | frequency step | Hz |
| `fo` | first frequency | Hz |

The frequency grid is `fo, fo+df, ..., fo+(nom-1)*df`, and it is the grid every
output lands on, and so choose it for the band you want: `fo = df = 0.0005` with
`nom = 200` gives 0.0005–0.1 Hz, i.e., 10–2000 s.

The half-space is the layer with `d = 0`, and must come last.

### 2. `event.specs`

Needed by `gram` and `green`.

**Source**

| symbol | meaning | units |
|---|---|---|
| `sig` | strike (clockwise from North) | deg |
| `del` | dip (from horizontal) | deg |
| `gam` | rake (in the fault plane, from the strike direction to the up-dip direction) | deg |
| `smoment` | scalar seismic moment | dyne-cm |
| `d0` | source depth (should match the model file) | km |
| `mzz`, `mxx`, `myy`, `mxz`, `myz`, `mxy` | moment tensor (North, East, Down). If any of the six is non-zero these are used and the fault angles are ignored. | |

**Receivers** — two ways to place them, chosen with `dist_calc_choice`:

| symbol | meaning |
|---|---|
| `dist_calc_choice` | `d` - receivers given directly as distances; `l` - spanned over a great circle between given lat/lon endpoints |
| `dist_reci`, `dist_recf`, `dx` | first and last receiver distance, and the step, km (`dist_calc_choice = d`) |
| `azimuth` (or `az`) | source-to-receiver azimuth, deg clockwise from North (`dist_calc_choice = d`) |
| `glatmin`, `glonmin`, `glatmax`, `glonmax` | lat/lon of the first and last receiver, deg (`dist_calc_choice = l`) |
| `nx` | number of receivers (`dist_calc_choice = l`) |
| `slatd`, `slond` | event lat/lon, deg |

**Band, modes, timing**

| symbol | meaning |
|---|---|
| `df`, `fo`, `nom` | frequency grid (should match the model file) |
| `mdmin`, `mdmx` | mode range to sum over |
| `to` | start-time offset applied to the synthetics, s |
| `jy`, `jd`, `jh`, `jm` | origin year, day-of-year, hour, minute |
| `evid`, `nz` | bookkeeping; used to name the seismogram file |

---

## The three commands

```
python3 SW1D.py disp   MODEL         [options]    dispersion curves and eigenfunctions
python3 SW1D.py gram   MODEL SPECS   [options]    synthetic seismograms
python3 SW1D.py green  MODEL SPECS   [options]    the 9-component Green's tensor
```

**Leave any of them off and you are asked for it** -- the command included --
so plain `python3 SW1D.py` is a complete invocation. The two ways of supplying
an input mix freely: whatever is on the command line is used, whatever is
missing is asked for, so `python3 SW1D.py --wave R` asks only for the command
and the model file.

At the prompt, `--wave` defaults to `both`; the command and the two file names
have no default and must be typed. One attempt each: a name that does not
exist, or a command or wave type that is not recognised, stops with the same
message the command line would have given.

Prompting needs a terminal. Run from a script, a pipeline or `cron` and a
missing argument is an error instead, quoting a command that would have worked
-- so nothing ever blocks waiting for input that cannot arrive.

`SW1D.py -h`, and `-h` on each command, carry the options, their units and
defaults, the files each command writes, and a worked example. Only the two
input-file layouts are left here.

### `disp`

Solves the eigenvalue problem and writes out dispersion files and eigenfunction files.

```bash
python3 SW1D.py disp mod.prem --wave both     # Rayleigh and Love
python3 SW1D.py disp mod.prem --wave R        # Rayleigh only
python3 SW1D.py disp mod.prem --wave love     # Love only
```

### `gram`

Sums the modes into R,T,Z displacement spectra at each receiver. The
mechanism, the moment and the receiver geometry all come from `event.specs`.

```bash
python3 SW1D.py gram mod.prem event.specs --wave both   # R, T and Z
python3 SW1D.py gram mod.prem event.specs --wave R      # Rayleigh: R and Z only
python3 SW1D.py gram mod.prem event.specs --wave L      # Love: T only
```

The output is a spectrum per component and receiver. Use
`plotting/plot_seismogram.py` to transform it into time series, with the option to convolve with a source time function.

### `green`

Builds the nine-component Green's tensor using Aki & Richards (7.146) and (7.147), in Cartesian components, for the source and receiver depths in the model file.

```bash
python3 SW1D.py green mod.prem event.specs --wave both
```

---

## Output files

Default output if binary; `--ascii` writes text instead.

| file (binary / ASCII) | written by | contents |
|---|---|---|
| `disp.<model>.<wave>.bin` / `disp.<model>.<wave>` | `disp` | dispersion curves and phase-velocity partial derivatives |
| `eigen.<model>.<wave>.bin` / `eigen.<model>.<wave>` | `disp` | eigenfunction depth profiles |
| `grid.<id>.bin` / `grid.<id>.ascii` | `gram` | R, T, Z displacement spectra |
| `green.bin` / `green.ascii` | `green` | all nine Green's-tensor components |
| `SW1D.log` | every run | what was read, computed and written |

`<wave>` is `ray` or `lov`.

### `disp.<model>.<wave>`

A header block — the number of model parameters (3: rho, Vp, Vs), the layer
count, the model printed as `thickness rho Vp Vs`, then the same table after flattening, followed by `N modes listed in file` and then, per mode and period:

```
mode  period(s)  c(km/s)  U(km/s)  Q  energy-balance-residual  nlayers  ipd
```

`ipd` is `1` when the three blocks of per-layer partial derivatives
(`dln c/dln rho`, `dln c/dln Vs`, `dln c/dln Vp`) follow, `0` when they do not.

### `eigen.<model>.<wave>`

The layer table is printed as `depth Vs rho Vp`, then for every mode:

```
mode  period(s)  c(km/s)  U(km/s)  (omega^2/(c*I3))  nlayers  flag
```

followed by one line per layer: `depth` and the eigenfunction columns. There is
one row per layer, down to the depth where the half-space starts;
below that the eigenfunction is carried analytically as an exponential tail.

### `grid.<id>.ascii`

The seismogram file. The name encodes the run as follows:

```
grid.<evid:4><nz:4><nreceivers:4><mdmin:1><mdmx:1>.ascii
grid.  0001    0000     0020        0        5    .ascii
```

`evid` and `nz` come from `event.specs` (`1` and `0` if absent); `mdmin` and
`mdmx` are the mode range.

```
line 1              nx  nom  3  <yyyy><ddd><hh><mm>
next nom lines      the frequencies, Hz
then                for each receiver, for each component R, T, Z:
                    nom complex values as Re Im pairs, 5 pairs per line
```

### `green.ascii`

One row per receiver and frequency:

```
irec  dist(km)  az(deg)  freq(Hz)  ReGxx ImGxx  ReGxy ImGxy  …  ReGzz ImGzz
```

Nine components in the order (xx, xy, xz, yx, yy, yz, zx, zy, zz), each as
a real and an imaginary part. The header line records the source and receiver
depths.

### `SW1D.log`

Every run appends a new block containign: the time, the exact command, both input paths, the
model and band summary, the wave types, the mode counts, and - for `gram` - the
event, source depth, mechanism, full moment tensor, start-time offset and
receiver geometry.

---

## Command reference

---

| flag | meaning |
|---|---|
| `MODEL` | positional: the model file; prompted for if omitted |
| `SPECS` | positional: the `event.specs` file, for `gram` and `green` only; prompted for if omitted |
| `--wave rayleigh\|love\|both` | `ray`/`R` and `lov`/`L` are accepted as shorthand; prompted for if omitted, default `both` |
| `--ensure-periods 20,50,100` | force a solution at these periods (seconds) |
| `--ascii` | write the outputs as ASCII text instead of the default binary |
| `-o`, `--outdir DIR` | where output goes, created if absent (default = current directory) |
| `-v`, `--verbose` | report each branch as the solver finds it |

`disp` takes no `SPECS`: the dispersion follows from the model file, and a
second file that could quietly move the band was more confusing than useful.

---

## Plotting

Four scripts in `plotting/`, each reading SW1D's own output. `--help` lists
every option.

Each writes a PNG by default — `dispersion.png`, `eigenfunctions.png`,
`green.png`, `seismogram.png`. `-o FILE` changes the name, and the extension
picks the format: `-o fig.pdf` gives vector output, in which case `--dpi` no
longer applies.

```bash
python3 plotting/plot_disp.py disp.mod.prem.ray.bin --y c --modes 0,1,2
python3 plotting/plot_disp.py disp.mod.prem.ray.bin disp.mod.prem.lov.bin

python3 plotting/plot_eigen.py eigen.mod.prem.ray.bin --period 20,50,100
python3 plotting/plot_eigen.py eigen.mod.prem.lov.bin --period 50 --comp UT,TT --zmax 400

python3 plotting/plot_seismogram.py grid.00010000002005.bin --specs event.specs --all
python3 plotting/plot_seismogram.py grid.00010000002005.bin --rec 1 --stf triangle --width 10

python3 plotting/plot_green.py green.ascii --rec 1
python3 plotting/plot_green.py green.ascii --dist 2000 --x period --part amp
```

Note: `--stf triangle` and `--stf gaussian` both require `--width`

---
