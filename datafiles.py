"""The seismogram and Green's-tensor output files."""
import numpy as np
import struct


FILE_TAG_GRID = b"PSGRID01"

FILE_TAG_GREEN = b"PSGREN02"

def _put(fh, a, dtype):
    """Append an array to an open binary file in the given dtype."""
    np.ascontiguousarray(a, dtype=dtype).tofile(fh)

def _get(fh, n, dtype):
    """Read n values of the given dtype from an open binary file."""
    return np.fromfile(fh, dtype=dtype, count=n)

def write_grid_ascii(s, path, evdate="0000000000"):
    """The grid.*.ascii text layout."""
    nx, nom = len(s["distance"]), len(s["frequency"])
    with open(path, "w") as fh:
        fh.write(f"{nx} {nom} 3 {evdate}\n")
        for f in s["frequency"]:
            fh.write(f"{f}\n")
        for i in range(nx):
            for nc in range(3):
                on_line = 0
                for k in range(nom):
                    v = s["spectra"][nc, i, k]
                    fh.write(f"{v.real:12.4E}{v.imag:12.4E}  ")
                    on_line += 1
                    if on_line == 5:
                        fh.write("\n")
                        on_line = 0
                if on_line:
                    fh.write("\n")

def write_grid_bin(s, path, evdate="0000000000"):
    """The seismogram spectra. Same content as grid.*.ascii, full precision."""
    nx, nom = len(s["distance"]), len(s["frequency"])
    ev = str(evdate).encode()
    with open(path, "wb") as fh:
        fh.write(FILE_TAG_GRID)
        fh.write(struct.pack("<4i", nx, nom, 3, len(ev)))
        fh.write(ev)
        fh.write(struct.pack("<3d", float(s["f0"]), float(s["df"]),
                             float(s["azimuth"])))
        _put(fh, s["frequency"], "<f8")
        _put(fh, s["distance"], "<f8")
        _put(fh, s["spectra"], "<c16")

def read_grid_bin(path):
    """-> dict with frequency, distance, azimuth, spectra[3, nrec, nfreq]."""
    with open(path, "rb") as fh:
        if fh.read(8) != FILE_TAG_GRID:
            raise ValueError(f"{path}: not a SW1D grid file")
        nx, nom, ncomp, nev = struct.unpack("<4i", fh.read(16))
        evdate = fh.read(nev).decode()
        f0, df, az = struct.unpack("<3d", fh.read(24))
        freq = _get(fh, nom, "<f8")
        dist = _get(fh, nx, "<f8")
        spec = _get(fh, ncomp*nx*nom, "<c16").reshape(ncomp, nx, nom)
    return dict(frequency=freq, distance=dist, azimuth=az, spectra=spec,
                f0=f0, df=df, evdate=evdate, components=("R", "T", "Z"))

def write_green_ascii(g, path):
    """Columns: receiver dist azimuth freq, then Re/Im of all 9 components."""
    names = [f"G{a}{b}" for a in "xyz" for b in "xyz"]
    with open(path, "w") as fh:
        fh.write(f"# SW1D Green's tensor, h={g['source_depth']} km, "
                 f"z={g['receiver_depth']} km\n")
        fh.write("# " + f"{'irec':>5}{'dist':>10}{'az':>9}{'freq':>14}")
        for n in names:
            fh.write(f"{'Re'+n:>16}{'Im'+n:>16}")
        fh.write("\n")
        for j in range(len(g["distance"])):
            for k, w in enumerate(g["omega"]):
                fh.write(f"{j:>7}{g['distance'][j]:>10.2f}"
                         f"{np.degrees(g['azimuth'][j]):>9.3f}"
                         f"{w/(2*np.pi):>14.8f}")
                for a in range(3):
                    for b in range(3):
                        v = g["G"][k, j, a, b]
                        fh.write(f"{v.real:>16.7E}{v.imag:>16.7E}")
                fh.write("\n")

def write_green_bin(g, path):
    """The Green's tensor. Same content as green.ascii, full precision."""
    nf, nr = len(g["omega"]), len(g["distance"])
    with open(path, "wb") as fh:
        fh.write(FILE_TAG_GREEN)
        fh.write(struct.pack("<2i", nf, nr))
        fh.write(struct.pack("<2d", float(g["source_depth"]),
                             float(g["receiver_depth"])))
        _put(fh, g["omega"], "<f8")
        _put(fh, g["distance"], "<f8")
        _put(fh, g["azimuth"], "<f8")
        for key in ("G", "G_rayleigh", "G_love"):
            _put(fh, g[key], "<c16")

def read_green_bin(path):
    """-> dict with omega, distance, azimuth, G[nfreq, nrec, 3, 3]."""
    with open(path, "rb") as fh:
        if fh.read(8) != FILE_TAG_GREEN:
            raise ValueError(f"{path}: not a SW1D Green's tensor file")
        nf, nr = struct.unpack("<2i", fh.read(8))
        h, z = struct.unpack("<2d", fh.read(16))
        om = _get(fh, nf, "<f8")
        dist = _get(fh, nr, "<f8")
        az = _get(fh, nr, "<f8")
        n = nf*nr*9
        G = _get(fh, n, "<c16").reshape(nf, nr, 3, 3)
        Gr = _get(fh, n, "<c16").reshape(nf, nr, 3, 3)
        Gl = _get(fh, n, "<c16").reshape(nf, nr, 3, 3)
    return dict(omega=om, distance=dist, azimuth=az, G=G, G_rayleigh=Gr,
                G_love=Gl, source_depth=h, receiver_depth=z)
