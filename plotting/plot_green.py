#!/usr/bin/env python3
"""Plot the SW1D Green's tensor from a green.ascii file.

    python3 plot_green.py GREEN [options]

Examples
    plot_green.py green.ascii --rec 1                 # all 9, amplitude
    plot_green.py green.ascii --comp zz --part real+imag
    plot_green.py green.ascii --comp xz,zz --x t
    plot_green.py green.ascii --comp all --part amp --logy
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AX = "xyz"
NAMES = [a+b for a in AX for b in AX]          # xx xy xz yx ... zz


def read_green(path):
    """-> (irec[], dist[], az[deg], freq[], G[nrow, 3, 3] complex)."""
    if path.endswith(".bin"):
        import struct
        with open(path, "rb") as fh:
            if fh.read(8) != b"PSGREN02":
                raise SystemExit(f"{path}: not a SW1D Green's tensor file")
            nf, nr = struct.unpack("<2i", fh.read(8))
            struct.unpack("<2d", fh.read(16))
            om = np.fromfile(fh, "<f8", nf)
            dist = np.fromfile(fh, "<f8", nr)
            az = np.fromfile(fh, "<f8", nr)
            G = np.fromfile(fh, "<c16", nf*nr*9).reshape(nf, nr, 3, 3)
        irec = np.repeat(np.arange(nr), nf)
        return (irec, np.repeat(dist, nf), np.degrees(np.repeat(az, nf)),
                np.tile(om/(2*np.pi), nr),
                G.transpose(1, 0, 2, 3).reshape(nr*nf, 3, 3))
    rows = [ln.split() for ln in open(path)
            if ln.strip() and not ln.lstrip().startswith("#")]
    a = np.array([[float(x) for x in r] for r in rows])
    irec, dist, az, freq = a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3]
    v = a[:, 4:].reshape(len(a), 3, 3, 2)
    return irec, dist, az, freq, v[..., 0] + 1j*v[..., 1]


def to_time(g, freq):
    """Inverse-transform one component onto a causal time axis."""
    df = freq[1] - freq[0]
    ndc = int(round(freq[0]/df))
    nom = len(freq)
    n = 8*(ndc + nom - 1)
    full = np.zeros(n//2 + 1, dtype=complex)
    full[ndc:ndc+nom] = np.conj(g)          # irfft sums exp(+iwt); delay needs -
    dt = (1.0/df)/n
    return np.arange(n)*dt, np.fft.irfft(full, n=n)


def main():
    """Draw the chosen tensor components for one receiver."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("green", help="SW1D green.bin (or the .ascii form)")
    ap.add_argument("--comp", default="all",
                    help="all, or comma separated from xx,xy,xz,yx,...,zz")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rec", type=int,
                   help="receiver index as written in the file (default: first)")
    g.add_argument("--dist", type=float,
                   help="pick the nearest receiver instead, km")
    ap.add_argument("--x", default="f", choices=["f", "period", "t"],
                    help="abscissa, which is what the domain is: f = frequency "
                         "in Hz (default), period = s, t = the time domain")
    ap.add_argument("--part", default=None,
                    choices=["amp", "phase", "real", "imag", "real+imag"],
                    help="which part of the complex spectrum (default amp)")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("-o", "--out", default="green.png",
                    help="output file; the extension picks the format "
                         "(default green.png)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution; ignored for vector formats "
                         "such as .pdf and .svg")
    a = ap.parse_args()

    if a.x == "t":
        for name, given in (("--part", a.part is not None), ("--logy", a.logy)):
            if given:
                raise SystemExit(f"{name} does not apply to --x t: the "
                                 f"time-domain trace is real and signed")
    part = a.part or "amp"
    if a.logy and part != "amp":
        raise SystemExit(f"--logy needs --part amp: {part} is signed")

    irec, dist, az, freq, G = read_green(a.green)
    uniq = sorted(set(irec))
    if a.dist is not None:
        # argmin always returns something, so check the range first
        lo, hi = dist.min(), dist.max()
        if not lo <= a.dist <= hi:
            raise SystemExit(f"--dist {a.dist:g}: the file has receivers "
                             f"from {lo:g} to {hi:g} km")
        pick = irec[np.argmin(np.abs(dist - a.dist))]
    else:
        pick = a.rec if a.rec is not None else uniq[0]
    if pick not in uniq:
        raise SystemExit(f"receiver {pick} not in file (have {uniq})")
    m = irec == pick
    f, g, d, ph = freq[m], G[m], dist[m][0], az[m][0]   # az is already deg
    k = np.argsort(f)
    f, g = f[k], g[k]

    comps = (NAMES if a.comp.lower() == "all"
             else [c.strip().lower() for c in a.comp.split(",")])
    for c in comps:
        if c not in NAMES:
            raise SystemExit(f"--comp {c}: expected one of {', '.join(NAMES)}")

    if len(comps) == 9:
        fig, axes = plt.subplots(3, 3, figsize=(13, 9), squeeze=False)
        cells = [(i, j) for i in range(3) for j in range(3)]
    else:
        n = len(comps)
        ncol = min(n, 3)
        nrow = int(np.ceil(n/ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6*ncol, 3.2*nrow),
                                 squeeze=False)
        cells = [(i//ncol, i % ncol) for i in range(n)]
        for i in range(n, nrow*ncol):
            axes[i//ncol][i % ncol].axis("off")

    for c, (r, col) in zip(comps, cells):
        ax = axes[r][col]
        i, j = AX.index(c[0]), AX.index(c[1])
        gij = g[:, i, j]

        if a.x == "t":
            t, y = to_time(gij, f)
            ax.plot(t, y, "-", color="darkviolet", alpha=0.6, lw=1.3)
            ax.set_xlabel("Time (s)")
        else:
            xv = f if a.x == "f" else 1.0/f
            if part == "amp":
                ax.plot(xv, np.abs(gij), "-", color="darkviolet", alpha=0.6, lw=1.4)
            elif part == "phase":
                ax.plot(xv, np.degrees(np.angle(gij)), "-", color="darkviolet", alpha=0.6, lw=1.4)
            elif part == "real":
                ax.plot(xv, gij.real, "-", color="darkviolet", alpha=0.6, lw=1.4)
            elif part == "imag":
                ax.plot(xv, gij.imag, "-", color="darkviolet", alpha=0.6,
                        lw=1.4)
            else:
                ax.plot(xv, gij.real, "-", color="darkviolet", alpha=0.6, lw=1.4, label="Re")
                ax.plot(xv, gij.imag, "r--", lw=1.2, label="Im")
                if (r, col) == cells[0]:
                    ax.legend(fontsize="small", frameon=False)
            ax.set_xlabel("Frequency (Hz)" if a.x == "f" else "Period (s)")

        # a zero component is a result (Love-only), and a log axis cannot show it
        yy = np.concatenate([l.get_ydata() for l in ax.get_lines()]) \
            if ax.get_lines() else np.array([0.0])
        log_ok = a.logy and bool(np.any(yy > 0))
        if log_ok:
            ax.set_yscale("log")

        ax.set_ylabel(f"G$_{{{c}}}$")
        ax.grid(True, alpha=0.25, ls=":")
        if not log_ok:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2),
                                useOffset=False)

    what = "time series" if a.x == "t" else part
    fig.text(0.5, 0.005, f"receiver {pick}:  r = {d:g} km,  "
             f"azimuth = {ph:g}$^\\circ$,  {what}", ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}  ({len(comps)} component(s), "
          f"{'time' if a.x == 't' else 'frequency'} domain)")


if __name__ == "__main__":
    main()
