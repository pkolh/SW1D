"""The command line: argument parsing, logging, and the three subcommands."""
import argparse
import datetime
import numpy as np
import os
import sys

from driver import solve
from rec_geometry import specs_azimuth, specs_geometry
from inputs import MT_KEYS, apply_specs, read_model, read_specs, specs_source, specs_t0
from modefiles import parse_disp, parse_eigen, write_modefile
from datafiles import write_green_ascii, write_green_bin, write_grid_ascii, write_grid_bin
from seismogram import seismograms
from greens import green_tensor

WAVES = {"r": "rayleigh", "ray": "rayleigh", "rayleigh": "rayleigh",
         "l": "love", "lov": "love", "love": "love", "both": "both"}

def _wave(s):
    """argparse type: accept rayleigh/love/both, and R and L as shorthand."""
    try:
        return WAVES[s.strip().lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"{s!r}: expected rayleigh, love or both (R and L also work)")

COMMANDS = {"disp": "disp", "dispersion": "disp",
            "gram": "gram", "seismogram": "gram",
            "green": "green"}

def _cmd(s):
    """Accept a command name or one of its aliases, and canonicalise it."""
    try:
        return COMMANDS[s.strip().lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"{s!r}: expected disp, gram or green")

TOP_EPILOG = """
two inputs, both plain text:

  MODEL   the layered model, the frequency band, and the source and receiver
          depths                                     example: examples/mod.prem
  SPECS   event.specs: the source mechanism and the receivers, for gram and
          green                                   example: examples/event.specs

Leave any of them off -- the command included -- and it is asked for at the
prompt, so plain `SW1D.py` is a complete command. The two can be mixed: what
is on the line is used, what is missing is asked for.

  SW1D.py disp  mod.prem --wave both
  SW1D.py gram  mod.prem event.specs --wave both
  SW1D.py green mod.prem event.specs --wave both
  SW1D.py                                   ask for everything
  SW1D.py --wave R                          ask for the command and the model

written to the output directory (-o, default the current one):

  disp.<model>.<ray|lov>    dispersion curves, Q and gamma        disp
  eigen.<model>.<ray|lov>   eigenfunction depth profiles          disp
  grid.<run id>             displacement spectra                  gram
  green                     the 9-component tensor                green
  SW1D.log                  appended once per run                 all three

Everything the run needs comes from the two files: there is no flag that
repeats what a file already says. README.md has both layouts.
"""

DISP_EPILOG = """
Always writes both, one pair per wave type, named after the model file:

  disp.<model>.ray    disp.<model>.lov     phase and group velocity, Q, gamma
  eigen.<model>.ray   eigen.<model>.lov    depth profiles

.bin unless --ascii is given.

  SW1D.py disp mod.prem --wave both
  SW1D.py disp mod.prem --wave R --ensure-periods 20,50,100
"""

GRAM_EPILOG = """
The output is SPECTRA, not time series: plotting/plot_seismogram.py does the
transform and can convolve a source time function.

  grid.<evid><nz><receivers><mdmin><mdmx>[.bin|.ascii]

--wave R gives the R and Z components, L gives T, both gives all three.
exp(-gamma r) is always applied along the path, gamma coming from the Q
columns of the model file; set those to 0 for an elastic run.

  SW1D.py gram mod.prem event.specs --wave both
"""

GREEN_EPILOG = """
Written as green.bin, or green.ascii with --ascii. That name carries no run
id, so a second green run in the same output directory OVERWRITES the first.

Components in the order xx xy xz yx yy yz zx zy zz, Cartesian in
(North, East, Up), for the source and receiver depths the model file sets.

  SW1D.py green mod.prem event.specs --wave both
"""

def _common_args(p, wants_specs=False):
    """Arguments shared by disp, gram and green."""
    p.add_argument("model", metavar="MODEL", nargs="?", default=None,
                   help="the layered model, the frequency band, and the "
                        "source and receiver depths; asked for at the prompt "
                        "if omitted (README.md has the layout)")
    if wants_specs:
        p.add_argument("specs", metavar="SPECS", nargs="?", default=None,
                       help="event.specs: the source mechanism and the "
                            "receivers; asked for at the prompt if omitted "
                            "(README.md has the key list)")
    p.add_argument("--wave", type=_wave, default=None,
                   metavar="{rayleigh,love,both}",
                   help="which wave types to solve: rayleigh (also ray, R) "
                        "gives the R and Z components, love (lov, L) gives T, "
                        "both gives all three; asked for at the prompt if "
                        "omitted, where the default is both")
    p.add_argument("--ensure-periods", metavar="T1,T2,...",
                   help="periods in seconds, comma separated, at which a "
                        "solution is forced even where the frequency grid "
                        "steps past them, e.g. 20,50,100")
    p.add_argument("--ascii", action="store_true",
                   help="write the original text layouts instead of the "
                        "default binary; binary is about half the size and "
                        "keeps full float64, text rounds to 5-7 significant "
                        "digits")
    p.add_argument("-o", "--outdir", default=".", metavar="DIR",
                   help="where the output files go, created if it does not "
                        "exist (default: the current directory)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="report each branch as the solver finds it")

# everything printed goes to SW1D.log too; some things go only there
LINES = []

def log(msg="", show=True):
    """Print and record a line; show=False records without printing."""
    if show:
        print(msg)
    LINES.append(msg)

def note(msg):
    """Record a line in the log without printing it."""
    log(msg, show=False)

def save_log(outdir):
    """Append this run's log lines to SW1D.log in outdir."""
    path = os.path.join(outdir, "SW1D.log")
    with open(path, "a") as fh:
        fh.write("\n" + "="*70 + "\n")
        fh.write("\n".join(LINES) + "\n")
    print(f"  wrote {path}")

def _die(msg):
    """Abort with a one-line error message."""
    raise SystemExit(f"SW1D: error: {msg}")

def _ask(what, prompt, example, default=None, coerce=None):
    """Ask for a missing input, or say how to supply it and stop."""
    if not sys.stdin.isatty():
        _die(f"no {what} given. Try: {example}")
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nSW1D: cancelled")
    if not answer:
        if default is None:
            _die(f"no {what} given. Try: {example}")
        answer = default
    if coerce is None:
        return answer
    try:
        return coerce(answer)
    except (argparse.ArgumentTypeError, ValueError) as e:
        _die(e)

def _read(reader, path, what):
    """Read a file with `reader`; any failure becomes a one-line error."""
    if not os.path.exists(path):
        _die(f"no such {what}: {path}")
    try:
        return reader(path)
    except SystemExit:
        raise
    except Exception as e:
        _die(f"could not read the {what} {path}: {e}")

def _cfg(a):
    """RunConfig from the model file, overridden by event.specs, then by flags."""
    cfg = _read(read_model, a.model, "model file")
    path = getattr(a, "specs", None)
    specs = _read(read_specs, path, "specs file") if path else None
    if specs:
        cfg = apply_specs(cfg, specs)
    if getattr(a, "ensure_periods", None):
        try:
            cfg.ensure_periods = tuple(float(x)
                                       for x in a.ensure_periods.split(","))
        except ValueError:
            _die(f"--ensure-periods {a.ensure_periods}: expected periods in "
                 f"seconds, comma separated, e.g. 20,50,100")
        if any(v <= 0 for v in cfg.ensure_periods):
            _die("--ensure-periods: periods must be positive")
    # event.specs can override what the model file said, so re-check
    try:
        cfg.validate()
    except ValueError as e:
        _die(e)
    return cfg, specs

def _solve(cfg, a):
    """Solve the requested wave types and write the mode files."""
    ray = lov = None
    # only disp writes mode files, and it always writes both
    eig = want = (a.cmd == "disp")
    base = os.path.basename(a.model)
    for wave, tag in (("rayleigh", "ray"), ("love", "lov")):
        if a.wave not in (wave, "both"):
            continue
        m, disp_text, eigen_text = solve(cfg, wave, verbose=a.verbose,
                                         eigenfunctions=eig, mode_files=want)
        log(f"  {m}")
        if wave == "rayleigh":
            ray = m
        else:
            lov = m
        if want:
            _write_mode_files(a, disp_text, eigen_text, wave,
                              f"{base}.{tag}", eig)
    return ray, lov

def _write_mode_files(a, disp_text, eigen_text, wave, stem, eig):
    """Write disp.* and eigen.*, binary or text."""
    jobs = [("disp", disp_text, parse_disp)]
    if eig:
        jobs.append(("eigen", eigen_text, parse_eigen))
    for kind, text, parse in jobs:
        if text is None:
            continue
        if a.ascii:
            p = os.path.join(a.outdir, f"{kind}.{stem}")
            open(p, "w").write(text)
        else:
            p = os.path.join(a.outdir, f"{kind}.{stem}.bin")
            write_modefile(parse(text, wave), p)
        log(f"  wrote {p}")

def _geometry(specs):
    """Receiver distances (km) and azimuth (deg), from event.specs."""
    try:
        return specs_geometry(specs)
    except ValueError as e:
        _die(e)

def _az_source(specs):
    """Where the azimuth came from, for the log."""
    try:
        return ("given" if specs_azimuth(specs) is not None
                else "from the glat/glon corners")
    except ValueError as e:
        _die(e)

def _moment(specs):
    """Moment tensor from the specs file, as a 3x3 in (North, East, Down)."""
    try:
        return specs_source(specs)
    except ValueError as e:
        _die(e)

def _log_source(specs, mt, dist, az):
    """Log the event, source mechanism and receiver geometry."""
    given = any(float(specs.get(k, 0.0)) != 0.0 for k in MT_KEYS)
    note(f"event {int(specs.get('evid', 1))}, "
         f"{int(specs.get('jy', 0)):04d}/{int(specs.get('jd', 0)):03d} "
         f"{int(specs.get('jh', 0)):02d}:{int(specs.get('jm', 0)):02d}")
    if "slatd" in specs and "slond" in specs:
        note(f"epicentre: {specs['slatd']} deg N, {specs['slond']} deg E")
    else:
        note("epicentre: not given (the azimuth was, so none was needed)")
    note(f"source depth: {specs.get('d0')} km")
    if given:
        note("mechanism: moment tensor given directly in the specs file")
    else:
        note(f"mechanism: strike {specs.get('sig')}, dip {specs.get('del')}, "
             f"rake {specs.get('gam')} deg")
    note(f"moment: {float(specs['smoment']):g} dyne-cm")
    note("moment tensor (North, East, Down), dyne-cm:")
    for row in mt:
        note("   " + "".join(f"{v:14.4E}" for v in row))
    note(f"start-time offset: {specs_t0(specs):g} s")
    note(f"receivers: {len(dist)}, {dist[0]:g} to {dist[-1]:g} km, "
         f"azimuth {az:g} deg ({_az_source(specs)})")

def main(argv=None):
    """Parse the command line, run the requested subcommand, write the outputs."""
    raw = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(
        prog="SW1D", formatter_class=raw,
        description="Surface-wave modes, seismograms and Green's tensors.",
        epilog=TOP_EPILOG)
    # not required: a missing command is asked for, like every other input
    sub = ap.add_subparsers(dest="cmd", required=False, metavar="COMMAND")

    p = sub.add_parser("disp", aliases=["dispersion"], formatter_class=raw,
                       help="dispersion curves and eigenfunctions",
                       description="Dispersion curves and eigenfunctions, "
                                   "from the model file alone.",
                       epilog=DISP_EPILOG)
    _common_args(p)

    p = sub.add_parser("gram", aliases=["seismogram"], formatter_class=raw,
                       help="synthetic seismograms",
                       description="Synthetic seismograms: the modes summed "
                                   "into displacement spectra.",
                       epilog=GRAM_EPILOG)
    _common_args(p, wants_specs=True)

    p = sub.add_parser("green", formatter_class=raw,
                       help="9-component Green's tensor",
                       description="The 9-component surface-wave Green's "
                                   "tensor, Aki & Richards (7.146)/(7.147).",
                       epilog=GREEN_EPILOG)
    _common_args(p, wants_specs=True)

    # the command must be settled before parse_args; -h is exempt
    argv = sys.argv[1:] if argv is None else list(argv)
    helping = any(t in ("-h", "--help") for t in argv)
    if not helping and not any(t in COMMANDS for t in argv):
        argv.insert(0, _ask("command", "command [disp/gram/green]: ",
                            "SW1D.py disp mod.prem --wave both", coerce=_cmd))

    a = ap.parse_args(argv)
    if a.cmd is None:              # e.g. `-o green`: consumed as a value
        a = ap.parse_args([_ask(
            "command", "command [disp/gram/green]: ",
            "SW1D.py disp mod.prem --wave both", coerce=_cmd)] + argv)
    a.cmd = COMMANDS[a.cmd]

    ex = f"SW1D.py {a.cmd} mod.prem"
    if a.cmd != "disp":
        ex += " event.specs"
    if a.model is None:
        a.model = _ask("model file", "model file: ", f"{ex} --wave both")
    if a.cmd != "disp" and a.specs is None:
        a.specs = _ask("specs file", "event.specs: ", f"{ex} --wave both")
    if a.wave is None:
        a.wave = _ask("wave type", "wave types [both]: ", f"{ex} --wave both",
                      default="both", coerce=_wave)
    try:
        os.makedirs(a.outdir, exist_ok=True)
    except OSError as e:
        _die(f"could not create the output directory {a.outdir}: {e}")
    if not os.access(a.outdir, os.W_OK):
        _die(f"the output directory {a.outdir} is not writable")
    cfg, specs = _cfg(a)
    note(f"SW1D {a.cmd}    "
         f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    note("command: " + " ".join(sys.argv))
    note(f"model file: {os.path.abspath(a.model)}")
    sp = getattr(a, "specs", None)
    note(f"specs file: {os.path.abspath(sp) if sp else '(none)'}")
    log(cfg.describe())
    note(f"waves: {a.wave}    output: "
         f"{'text' if a.ascii else 'binary'}")
    ray, lov = _solve(cfg, a)

    if a.cmd == "gram":
        dist, az = _geometry(specs)
        mt = _moment(specs)
        _log_source(specs, mt, dist, az)
        s = seismograms(ray, lov, mt, dist, azimuth=az, cfg=cfg,
                        t0=specs_t0(specs))
        # output name encodes the run: grid.<evid><nz><nx><mdmin><mdmx>
        ev = "0000000000"
        evid, nz = 1, 0
        if specs:
            ev = (f"{int(specs.get('jy',0)):04d}{int(specs.get('jd',0)):03d}"
                  f"{int(specs.get('jh',0)):02d}{int(specs.get('jm',0)):02d}")
            evid, nz = int(specs.get("evid", 1)), int(specs.get("nz", 0))
        stem = (f"grid.{evid:04d}{nz:04d}{len(s['distance']):04d}"
                f"{cfg.mode_min:01d}{cfg.mode_max:01d}")
        if a.ascii:
            p = os.path.join(a.outdir, stem + ".ascii")
            write_grid_ascii(s, p, evdate=ev)
        else:
            p = os.path.join(a.outdir, stem + ".bin")
            write_grid_bin(s, p, evdate=ev)
        log(f"  wrote {p}")

    elif a.cmd == "green":
        dist, az = _geometry(specs)
        note(f"receivers: {len(dist)}, {dist[0]:g} to {dist[-1]:g} km, "
             f"azimuth {az:g} deg ({_az_source(specs)})")
        g = green_tensor(ray, lov, distance=dist, azimuth=az,
                         omega=2*np.pi*cfg.frequencies)
        if a.ascii:
            p = os.path.join(a.outdir, "green.ascii")
            write_green_ascii(g, p)
        else:
            p = os.path.join(a.outdir, "green.bin")
            write_green_bin(g, p)
        log(f"  Green's tensor: {len(g['omega'])} freqs, "
            f"{len(g['distance'])} receivers, h={g['source_depth']} km, "
            f"z={g['receiver_depth']} km")
        log(f"  wrote {p}")

    save_log(a.outdir)
    return 0
