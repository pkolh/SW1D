#!/usr/bin/env python3
"""Plot SW1D eigenfunction depth profiles from an eigen.<model>.<wave> file.

    python3 plot_eigen.py EIGEN [options]

Examples
    plot_eigen.py eigen.mod.prem.ray.bin --period 20,50,100
    plot_eigen.py eigen.mod.prem.ray.bin --modes 0,1,2 --period 50
    plot_eigen.py eigen.mod.prem.lov.bin --period 50 --comp UT,TT --zmax 400
    plot_eigen.py eigen.mod.prem.ray --period 50 --norm        # the text form

Depths are as the file carries them, so a run with flattening on plots the
flattened depths.
"""
import argparse
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the columns after depth, in file order
COLUMNS = {"rayleigh": ["UZ", "UR", "TZ", "TR"], "love": ["UT", "TT"]}
DEFAULT = {"rayleigh": ["UZ", "UR"], "love": ["UT"]}



def mathlabel(name):
    """Component name as a maths label: UZ -> U$_Z$."""
    return f"{name[0]}$_{{{name[1:]}}}$"


def read_eigen(path):
    """-> dict with wave, mode, period, c, u and a list of (ls, ncol) profiles."""
    with open(path, "rb") as fh:
        if fh.read(8) == b"PSEIGN01":
            wave, nlayer, _nbr, nrec, nbranch, ncol = \
                struct.unpack("<6i", fh.read(24))
            neig, = struct.unpack("<q", fh.read(8))
            nb, = struct.unpack("<i", fh.read(4))
            fh.read(nb)
            np.fromfile(fh, "<f8", nlayer*4)                 # the model table
            np.fromfile(fh, "<i4", 4*nbranch)                # branch bookkeeping
            mode, ls, _int = (np.fromfile(fh, "<i4", nrec) for _ in range(3))
            per, c, u, _calc = (np.fromfile(fh, "<f8", nrec) for _ in range(4))
            off = np.fromfile(fh, "<i8", nrec + 1)
            eig = np.fromfile(fh, "<f8", neig)
            prof = [eig[off[k]:off[k+1]].reshape(int(ls[k]), ncol)
                    for k in range(nrec)]
            return dict(wave="rayleigh" if wave == 1 else "love", mode=mode,
                        period=per, c=c, u=u, profile=prof)

    return _read_eigen_text(path)


def _read_eigen_text(path):
    """The text layout: a banner, the model, then one block per mode."""
    lines = open(path).read().splitlines()
    nlayer = int(lines[1].split()[0])
    i = 3 + nlayer                                # banner, count, model, "N modes"

    mode, per, c, u, prof = [], [], [], [], []
    while i < len(lines):
        if lines[i].rstrip().endswith("mode number"):
            i += 2                                # the branch header and its counts
            continue
        p = lines[i].split()
        mode.append(int(p[0])); per.append(float(p[1]))
        c.append(float(p[2])); u.append(float(p[3]))
        ls = int(p[5])
        i += 1
        prof.append(np.array([[float(x) for x in lines[i + r].split()]
                              for r in range(ls)]))
        i += ls
    wave = "love" if os.path.basename(path).split(".")[-1].startswith("lov") \
        else "rayleigh"
    return dict(wave=wave, mode=np.array(mode), period=np.array(per),
                c=np.array(c), u=np.array(u), profile=prof)


def pick(d, branch, period):
    """The record for one branch nearest a wanted period; -> index or None."""
    m = np.flatnonzero(d["mode"] == branch)
    if not len(m):
        return None
    return m[np.argmin(np.abs(d["period"][m] - period))]


def main():
    """Draw the chosen components against depth, one panel per component."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eigen", help="SW1D eigen.*.bin (or the text form)")
    ap.add_argument("--period", default="50",
                    help="comma separated periods in s; the nearest solved "
                         "period is used (default 50)")
    ap.add_argument("--modes", default="0",
                    help="comma separated branch numbers (default 0)")
    ap.add_argument("--comp", default=None,
                    help="comma separated columns; Rayleigh has "
                         "UZ,UR,TZ,TR and Love UT,TT. Default: the "
                         "displacements alone")
    # both set the depth axis, so asking for both is a contradiction
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--zmax", type=float,
                   help="deepest depth to draw, km. Without it the axis stops "
                        "just below where the curves die away")
    g.add_argument("--full", action="store_true",
                   help="draw the whole model instead, halfspace included")
    ap.add_argument("--norm", action="store_true",
                    help="scale each curve to unit maximum")
    ap.add_argument("-o", "--out", default="eigenfunctions.png",
                    help="output file; the extension picks the format "
                         "(default eigenfunctions.png)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution; ignored for vector formats "
                         "such as .pdf and .svg")
    a = ap.parse_args()

    d = read_eigen(a.eigen)
    names = COLUMNS[d["wave"]]
    comps = DEFAULT[d["wave"]] if a.comp is None else \
        [c.strip().upper() for c in a.comp.split(",")]
    for c in comps:
        if c not in names:
            raise SystemExit(f"--comp {c}: a {d['wave']} file has "
                             f"{', '.join(names)}")
    periods = [float(p) for p in a.period.split(",")]
    branches = [int(m) for m in a.modes.split(",")]

    ncol = len(comps)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7*ncol, 5.6), squeeze=False,
                             sharey=True)
    drawn, floor = [], 0.0
    for ci, comp in enumerate(comps):
        ax = axes[0][ci]
        j = names.index(comp) + 1                 # column 0 is depth
        for bi, b in enumerate(branches):
            for pi, want in enumerate(periods):
                k = pick(d, b, want)
                if k is None:
                    continue
                z, y = d["profile"][k][:, 0], d["profile"][k][:, j]
                if a.norm and np.max(np.abs(y)) > 0:
                    y = y/np.max(np.abs(y))
                if a.zmax is not None:
                    m = z <= a.zmax
                    z, y = z[m], y[m]
                ax.plot(y, z, lw=1.4, color=f"C{(bi*len(periods) + pi) % 10}",
                        label=(f"mode {b}, T = {d['period'][k]:.1f} s"
                               if ci == 0 else None))
                drawn.append((b, d["period"][k]))
                # drawing the whole model squashes the curves into the top 5%
                big = np.flatnonzero(np.abs(y) > 0.01*np.max(np.abs(y)))
                if len(big):
                    floor = max(floor, z[big[-1]])
        ax.axvline(0.0, color="gray", lw=0.8, alpha=0.6)
        ax.set_xlabel(mathlabel(comp))
        ax.grid(True, alpha=0.25, ls=":")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2),
                            useOffset=False)
    if not drawn:
        raise SystemExit(f"{a.eigen}: no record for branch(es) "
                         f"{', '.join(map(str, branches))}")
    axes[0][0].set_ylabel("Depth (km)")
    top, bottom = axes[0][0].get_ylim()
    if a.zmax is None and not a.full and floor > 0:
        bottom = 1.15*floor
    axes[0][0].set_ylim(max(bottom, top), 0.0)
    axes[0][0].legend(fontsize="small", frameon=False)
    fig.text(0.5, 0.005, f"{os.path.basename(a.eigen)}  ({d['wave']})",
             ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}  ({len(drawn)} curve(s), "
          f"{', '.join(comps)}, periods "
          + ", ".join(f"{p:.1f}" for p in sorted({t for _, t in drawn})) + " s)")


if __name__ == "__main__":
    main()
