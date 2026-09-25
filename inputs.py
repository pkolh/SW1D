"""Reading the two input files: the model file and event.specs."""
import numpy as np
import os

from config import RunConfig
from source import double_couple

MT_KEYS = ("mzz", "mxx", "myy", "mxz", "myz", "mxy")

def _four_numbers(line):
    """True if the line starts with four numbers, i.e. is 'c1 c2 nbran1 nbran2'."""
    p = line.split()
    if len(p) < 4:
        return False
    try:
        [float(x) for x in p[:4]]
    except ValueError:
        return False
    return True

def read_layers(path):
    """A 6-column layer table: d vp vs rho Qbeta Qalpha. -> (6, n) columns."""
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            rows.append([float(x) for x in line.split()[:6]])
    return np.array(rows).T

def read_model(path):
    """Parse a model file. Returns a RunConfig."""
    lines = open(path).read().splitlines()
    it = iter(lines)

    def nxt(what):
        """Next non-blank line, naming what was expected if the file ends first."""
        while True:
            try:
                ln = next(it)
            except StopIteration:
                raise ValueError(f"the model file ends before {what}") from None
            if ln.strip():
                return ln

    n0, iefl, tref = nxt("the layer count line").split()[:3]
    n0, iefl, tref = int(float(n0)), int(float(iefl)), float(tref)

    if n0 < 1:
        raise ValueError(f"the model file declares {n0} layers")
    d, vp, vs, rho, qb, qa = [], [], [], [], [], []
    for i in range(n0):
        v = [float(x) for x in nxt(f"layer {i+1} of {n0}").split()[:6]]
        if len(v) < 6:
            raise ValueError(f"layer {i+1} has {len(v)} columns, expected 6: "
                             f"d vp vs rho Qbeta Qalpha")
        d.append(v[0]); vp.append(v[1]); vs.append(v[2])
        rho.append(v[3]); qb.append(v[4]); qa.append(v[5])

    # old files put a wave flag and a file name here; only the real line
    # starts with four numbers, so the others are skipped
    line = nxt("the c1 c2 nbran1 nbran2 line")
    if not _four_numbers(line):
        nxt("the c1 c2 nbran1 nbran2 line")          # the old binary file name
        line = nxt("the c1 c2 nbran1 nbran2 line")
    p = line.split()
    if len(p) < 4:
        raise ValueError("expected 'c1 c2 nbran1 nbran2', got " + " ".join(p))
    c1, c2, nbran1, nbran2 = float(p[0]), float(p[1]), int(p[2]), int(p[3])

    p = nxt("the nsrc nom df fo line").split()
    if len(p) < 4:
        raise ValueError("expected 'nsrc nom df fo', got " + " ".join(p))
    nsrc, nom, df, fo = int(p[0]), int(p[1]), float(p[2]), float(p[3])

    sdep = [float(x) for x in nxt("the source depths").split()[:nsrc]]
    if len(sdep) < nsrc:
        raise ValueError(f"the model file declares nsrc={nsrc} but lists "
                         f"{len(sdep)} source depth(s)")
    rdep = float(nxt("the receiver depth").split()[0])

    return RunConfig(thickness=d, vp=vp, vs=vs, rho=rho, qbeta=qb, qalpha=qa,
                     model_name=os.path.basename(path),
                     f0=fo, df=df, nfreq=nom,
                     source_depths=tuple(sdep), receiver_depth=rdep,
                     mode_min=nbran1, mode_max=nbran2, cmin=c1, cmax=c2,
                     flatten=bool(iefl), tref=tref)

def read_specs(path):
    """Parse an event.specs file: source, mechanism, receiver geometry, timing."""
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 2:
            continue
        k, v = p[0], p[1]
        try:
            out[k] = int(v) if v.lstrip("+-").isdigit() else float(v)
        except ValueError:
            out[k] = v
    return out

def _need(s, keys, what):
    """Raise if any of `keys` is missing from a parsed specs mapping."""
    missing = [k for k in keys if k not in s]
    if missing:
        raise ValueError("event.specs is missing "
                         + ", ".join(missing) + f" (needed for {what})")

def specs_source(s):
    """The moment tensor, a 3x3 array in (North, East, Down), dyne-cm."""
    _need(s, ("smoment",), "the source strength")
    m0 = float(s["smoment"])
    comps = [float(s.get(k, 0.0)) for k in MT_KEYS]
    if any(c != 0.0 for c in comps):
        mzz, mxx, myy, mxz, myz, mxy = comps
        return m0*np.array([[mxx, mxy, mxz],
                            [mxy, myy, myz],
                            [mxz, myz, mzz]])
    _need(s, ("sig", "del", "gam"), "the source mechanism")
    return double_couple(s["sig"], s["del"], s["gam"], m0=m0)

def specs_t0(s, default=0.0):
    """Start-time offset `to`, in seconds."""
    return float(s.get("to", default))

def apply_specs(cfg, s):
    """Override a model file-derived RunConfig with the fields that overlap."""
    for key, attr in (("fo", "f0"), ("df", "df"), ("nom", "nfreq"),
                      ("mdmin", "mode_min"), ("mdmx", "mode_max")):
        if key in s:
            setattr(cfg, attr, s[key])
    if "d0" in s:
        cfg.source_depths = (float(s["d0"]),)
    return cfg
