#!/usr/bin/env python3
"""Plot SW1D seismograms from a grid.* file.

    python3 plot_seismogram.py GRID [--specs event.specs] [options]

Examples
    plot_seismogram.py grid.00010000002005.bin --rec 8
    plot_seismogram.py grid.00010000002005.bin --specs event.specs --dist 2000 --comp Z
    plot_seismogram.py grid.ascii --specs event.specs --all --comp R
    plot_seismogram.py grid.ascii --rec 8 --stf triangle --width 8
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = ["R", "T", "Z"]


def read_grid(path):
    """-> (freqs[nom], spec[3, nx, nom] complex), from the binary or the text."""
    if path.endswith(".bin"):
        import struct
        with open(path, "rb") as fh:
            if fh.read(8) != b"PSGRID01":
                raise SystemExit(f"{path}: not a SW1D grid file")
            nx, nom, ncomp, nev = struct.unpack("<4i", fh.read(16))
            fh.read(nev)
            struct.unpack("<3d", fh.read(24))
            freq = np.fromfile(fh, "<f8", nom)
            np.fromfile(fh, "<f8", nx)
            spec = np.fromfile(fh, "<c16", ncomp*nx*nom).reshape(ncomp, nx, nom)
        return freq, spec
    tok = open(path).read().split()
    nx, nom = int(tok[0]), int(tok[1])
    f = np.array([float(x) for x in tok[4:4+nom]])
    v = np.array([float(x) for x in tok[4+nom:]])
    v = v.reshape(nx, 3, nom, 2)
    return f, (v[..., 0] + 1j*v[..., 1]).transpose(1, 0, 2)


def read_specs(path):
    """Parse event.specs into a dict; numeric where the value is numeric."""
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        p = line.split()
        if len(p) >= 2:
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                out[p[0]] = p[1]
    return out


def distances_from(specs, nx):
    """Receiver distances in km, or indices when the specs file has none."""
    if specs and all(k in specs for k in ("dist_reci", "dx")):
        return specs["dist_reci"] + specs["dx"]*np.arange(nx), True
    return np.arange(1, nx+1, dtype=float), False


def stf_spectrum(kind, width, f):
    """Spectrum of a unit-area source time function."""
    if kind == "none" or not width:
        return np.ones_like(f)
    w = np.pi*f*width
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "triangle":                     # base `width`
            s = (np.sin(w/2)/(w/2))**2
        elif kind == "gaussian":                   # FWHM ~ width
            s = np.exp(-(np.pi*f*width)**2/(4*np.log(2)))
        else:
            raise SystemExit(f"unknown --stf {kind}")
    return np.where(f == 0, 1.0, s)


def to_time(spec, f0, df, nom, oversample, stf, width):
    """Displacement trace from one channel/receiver spectrum."""
    ndc = int(round(f0/df))
    n = 2*(ndc + nom - 1)*oversample
    full = np.zeros(n//2 + 1, dtype=complex)
    full[ndc:ndc+nom] = np.conj(spec)
    dt = (1.0/df)/n
    if stf != "none":
        full *= stf_spectrum(stf, width, np.fft.rfftfreq(n, dt))
    return np.arange(n)*dt, n*np.fft.irfft(full, n=n)


def main():
    """Inverse-transform the spectra and draw receiver x component."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", help="SW1D grid.*.bin (or the .ascii form)")
    ap.add_argument("--specs", help="event.specs, for receiver distances")
    ap.add_argument("--comp", default="all",
                    help="R, T, Z, all, or a combination like RZ (default all)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rec", type=int, help="receiver number, 1-based")
    g.add_argument("--dist", type=float, help="pick the nearest receiver, km")
    g.add_argument("--all", action="store_true", help="every receiver, as a grid")
    ap.add_argument("--stf", default="none",
                    choices=["none", "tri", "triangle", "gauss", "gaussian"],
                    help="source time function; needs --width")
    ap.add_argument("--width", type=float, help="STF width, s: triangle base, "
                                                "or Gaussian FWHM")
    ap.add_argument("--over", type=int, default=4,
                    help="oversampling factor for the inverse transform")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--tmax", type=float)
    ap.add_argument("-o", "--out", default="seismogram.png",
                    help="output file; the extension picks the format "
                         "(default seismogram.png)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution; ignored for vector formats "
                         "such as .pdf and .svg")
    a = ap.parse_args()

    stf = {"tri": "triangle", "gauss": "gaussian"}.get(a.stf, a.stf)
    if stf != "none" and not a.width:
        raise SystemExit(f"--stf {a.stf} needs --width SECONDS")
    if stf == "none" and a.width:
        raise SystemExit("--width does nothing without --stf")

    f, spec = read_grid(a.grid)
    nchan, nx, nom = spec.shape
    f0, df, fmax = f[0], f[1]-f[0], f[-1]
    specs = read_specs(a.specs) if a.specs else None
    dist, dist_in_km = distances_from(specs, nx)

    comps = (COMPONENTS if a.comp.lower() == "all"
             else [c for c in a.comp.upper() if c in COMPONENTS])
    if not comps:
        raise SystemExit(f"--comp {a.comp}: nothing to plot")

    if a.all:
        recs = list(range(1, nx+1))
    elif a.dist is not None:
        if not dist_in_km:
            raise SystemExit(
                "--dist needs receiver distances. Give --specs event.specs."
                if specs is None else
                "--dist needs receiver distances, and this specs file does not "
                "carry them: 'l' geometry has no dist_reci/dx. Use --rec.")
        lo, hi = dist.min(), dist.max()
        if not lo <= a.dist <= hi:
            raise SystemExit(f"--dist {a.dist:g}: the file has receivers from "
                             f"{lo:g} to {hi:g} km")
        recs = [int(np.argmin(np.abs(dist - a.dist))) + 1]
    else:
        recs = [a.rec or 1]
    for r in recs:
        if not 1 <= r <= nx:
            raise SystemExit(f"receiver {r} outside 1..{nx}")

    fig, axes = plt.subplots(len(recs), len(comps),
                             figsize=(4.6*len(comps), 2.3*len(recs) + 1.2),
                             squeeze=False, sharex=False)
    for i, rec in enumerate(recs):
        for j, comp in enumerate(comps):
            ax = axes[i][j]
            ci = COMPONENTS.index(comp)
            t, u = to_time(spec[ci, rec-1], f0, df, nom, a.over,
                           stf, a.width)
            ax.plot(t, u, "-", color="darkviolet", alpha=0.6, lw=1.4)
            lo = a.tmin if a.tmin is not None else t[0]
            hi = a.tmax if a.tmax is not None else t[-1]
            ax.set_xlim(lo, hi)
            ax.grid(True, alpha=0.25, ls=":")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            if i == 0:
                ax.set_title(f"U$_{comp}$", fontsize=12)
            if j == 0:
                lab = (f"[{dist[rec-1]:.0f} km]" if dist_in_km
                       else f"[rec {rec}]")
                ax.set_ylabel(f"{lab}\nDisplacement")
            if i == len(recs)-1:
                ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}  ({len(recs)} receiver(s) x {len(comps)} component(s))")


if __name__ == "__main__":
    main()
