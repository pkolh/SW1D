#!/usr/bin/env python3
"""Plot SW1D dispersion curves from a disp.<model>.<wave> file.

    python3 plot_disp.py DISP [DISP ...] [options]

Examples
    plot_disp.py disp.mod.prem.ray.bin                      # c and U, all modes
    plot_disp.py disp.mod.prem.ray.bin --y c --modes 0,1,2
    plot_disp.py disp.mod.prem.ray.bin disp.mod.prem.lov.bin # Rayleigh vs Love
    plot_disp.py disp.mod.prem.ray --y all --x f            # the text form
"""
import argparse
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANTITIES = {
    "c":     ("Phase velocity (km/s)", "c"),
    "u":     ("Group velocity (km/s)", "u"),
    "q":     ("Mode Q", "q"),
    "gamma": (r"$\gamma$ (1/km)", "gamma"),
    "flan":  (r"$s_2/s_1 - 1$", "flan"),
}
LINESTYLES = ["-", "--", ":", "-."]
STYLE_NAMES = ["solid", "dashed", "dotted", "dash-dot"]


def read_disp(path):
    """-> dict with mode, period, c, u, q, flan and the wave type."""
    with open(path, "rb") as fh:
        tag = fh.read(8)
        if tag == b"PSDISP01":
            wave, _npar, nlayer, _nbr, nrec, ndone = \
                struct.unpack("<6i", fh.read(24))
            nder, = struct.unpack("<q", fh.read(8))
            np.fromfile(fh, "<f8", 2*nlayer*4)          # model, flattened model
            np.fromfile(fh, "<i4", ndone)               # branch-done marks
            mode, ls, ipd = (np.fromfile(fh, "<i4", nrec) for _ in range(3))
            per, c, u, q, flan = (np.fromfile(fh, "<f8", nrec) for _ in range(5))
            return dict(wave="rayleigh" if wave == 1 else "love", mode=mode,
                        period=per, c=c, u=u, q=q, flan=flan)

    return _read_disp_text(path)


def _read_disp_text(path):
    """The text layout: two model tables, then one row per mode."""
    lines = open(path).read().splitlines()
    nlayer = int(lines[1].split()[0])
    i = 2 + nlayer
    if lines[i].strip() == "flattened model":
        i += 1 + nlayer
    i += 1                                              # "N modes listed in file"

    rows = []
    while i < len(lines):
        p = lines[i].split()
        i += 1
        if not p or p[0].startswith("*"):
            continue
        if len(p) < 8:                                  # a partial-derivative block
            continue
        rows.append([float(x) for x in p[:8]])
        if int(float(p[7])):                            # ipd: 3*ls values follow
            want, got = 3*int(float(p[6])), 0
            while got < want:
                got += len(lines[i].split())
                i += 1
    r = np.array(rows)
    # the text form has the same columns for both waves, so the name decides
    wave = "love" if os.path.basename(path).split(".")[-1].startswith("lov") \
        else "rayleigh"
    return dict(wave=wave, mode=r[:, 0].astype(int), period=r[:, 1],
                c=r[:, 2], u=r[:, 3], q=r[:, 4], flan=r[:, 5])


def gamma(d):
    """The spatial decay rate the solver derives from Q: w/(2 c Q), 1/km."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["q"] > 0, np.pi/(d["period"]*d["c"]*d["q"]), 0.0)


def main():
    """Read the dispersion file(s), draw one panel per quantity."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("disp", nargs="+",
                    help="SW1D disp.*.bin (or the text form); several are "
                         "overlaid, e.g. the Rayleigh and the Love file")
    ap.add_argument("--y", default="cu",
                    help="cu (default), all, or comma separated from "
                         + ", ".join(QUANTITIES))
    ap.add_argument("--x", default="period", choices=["period", "f"],
                    help="abscissa: period in s (default) or f in Hz")
    ap.add_argument("--modes", default="all",
                    help="comma separated branch numbers, e.g. 0,1,2")
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("-o", "--out", default="dispersion.png",
                    help="output file; the extension picks the format "
                         "(default dispersion.png)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution; ignored for vector formats "
                         "such as .pdf and .svg")
    a = ap.parse_args()

    keys = {"cu": ["c", "u"], "all": list(QUANTITIES)}.get(
        a.y.lower(), [k.strip().lower() for k in a.y.split(",")])
    for k in keys:
        if k not in QUANTITIES:
            raise SystemExit(f"--y {k}: expected one of {', '.join(QUANTITIES)}")

    files = [read_disp(p) for p in a.disp]
    for d in files:
        d["gamma"] = gamma(d)
    want = (None if a.modes.lower() == "all"
            else [int(m) for m in a.modes.split(",")])

    ncol = min(len(keys), 2)
    nrow = int(np.ceil(len(keys)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2*ncol, 4.0*nrow),
                             squeeze=False)
    for i in range(len(keys), nrow*ncol):
        axes[i//ncol][i % ncol].axis("off")

    for n, key in enumerate(keys):
        ax = axes[n//ncol][n % ncol]
        label, field = QUANTITIES[key]
        for fi, (d, path) in enumerate(zip(files, a.disp)):
            branches = sorted(set(d["mode"].tolist()))
            if want is not None:
                branches = [b for b in branches if b in want]
            for b in branches:
                m = d["mode"] == b
                x = d["period"][m] if a.x == "period" else 1.0/d["period"][m]
                k = np.argsort(x)
                ax.plot(x[k], d[field][m][k], LINESTYLES[fi % len(LINESTYLES)],
                        lw=1.3, color=f"C{b % 10}",
                        label=(f"mode {b}" if fi == 0 and n == 0 else None))
        ax.set_xlabel("Period (s)" if a.x == "period" else "Frequency (Hz)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25, ls=":")
        if a.logx:
            ax.set_xscale("log")
        if a.logy:
            ax.set_yscale("log")
    axes[0][0].legend(fontsize="small", frameon=False, ncol=2)

    fig.text(0.5, 0.005, "     ".join(
        f"{os.path.basename(p)} ({d['wave']}, "
        f"{STYLE_NAMES[i % len(STYLE_NAMES)]})"
        for i, (p, d) in enumerate(zip(a.disp, files))), ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(a.out, dpi=a.dpi)

    shown = sorted({b for d in files for b in set(d["mode"].tolist())
                    if want is None or b in want})
    print(f"wrote {a.out}  ({len(files)} file(s), branches "
          f"{', '.join(str(b) for b in shown)}, {len(keys)} panel(s))")


if __name__ == "__main__":
    main()
