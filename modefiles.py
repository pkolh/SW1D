"""The disp.* and eigen.* mode files, as text and as binary."""
import numpy as np
import struct

from common import common
from datafiles import _get, _put

# the header blocks at the top of the disp.* and eigen.* files

MODEL_FMT = "{:7.3f}" + "{:10.6f}"*3

def before_flat(iouf2, s10, common):
    """Write the model table that heads both the disp and eigen files."""
    iouf2.write("3 no. of model parameters for part. der.\n")
    iouf2.write(str(common.n) + " no. of layers in model\n")
    for i in range(common.n):
        iouf2.write(MODEL_FMT.format(common.d[i], common.ro[i],
                                     common.vps[i], common.vss[i]) + "\n")
    s10.write("eigenfunction file from SW1D\n")
    s10.write(str(common.n) + " no. of layers in model\n")
    depth = [0.0]
    for i in range(common.n):
        depth.append(depth[-1] + common.d[i])
        s10.write(MODEL_FMT.format(depth[i], common.vss[i],
                                   common.ro[i], common.vps[i]) + "\n")

def after_flat(iouf2, common):
    """Append the flattened model to the dispersion file."""
    iouf2.write(" flattened model \n")
    for i in range(common.n):
        iouf2.write(MODEL_FMT.format(common.d[i], common.ro[i],
                                     common.vps[i], common.vss[i]) + "\n")

def nbran_line(iouf2, s10, nbran):
    """Write the branch-count line into both mode files."""
    line = "{:3}".format(nbran) + " modes listed in file\n"
    iouf2.write(line)
    s10.write(line)

def write_dispersion(m, path):
    """Columns: mode period(s) frequency(Hz) c(km/s) u(km/s) gamma(1/km)."""
    with open(path, "w") as fh:
        fh.write(f"# {m.wave} dispersion from SW1D\n")
        fh.write(f"# {'mode':>5}{'period':>14}{'freq':>14}"
                 f"{'c':>12}{'u':>12}{'gamma':>14}\n")
        for n in m.branches():
            for i in m.sorted_branch(n):
                fh.write(f"{n:>7}{m.period[i]:>14.6f}{m.frequency[i]:>14.8f}"
                         f"{m.c[i]:>12.6f}{m.u[i]:>12.6f}"
                         f"{m.gamma[i]:>14.6e}\n")

def write_eigenfunctions(m, path):
    """Per mode: a header line, then depth and the eigenfunction columns."""
    if m.eig is None or all(e is None for e in m.eig):
        raise ValueError("solve(..., eigenfunctions=True) was not used")
    with open(path, "w") as fh:
        fh.write(f"# {m.wave} eigenfunctions from SW1D\n")
        for n in m.branches():
            for i in m.sorted_branch(n):
                e = m.eig[i]
                if e is None:
                    continue
                depth = np.concatenate([[0.0], np.cumsum(e[0])])[:-1]
                fh.write(f"{n:>4}{m.period[i]:>14.6f}"
                         f"{m.c[i]:>12.6f}{m.u[i]:>12.6f}{len(depth):>6}\n")
                for j in range(len(depth)):
                    fh.write(f"{depth[j]:>14.6E}")
                    for k in range(1, e.shape[0]):
                        fh.write(f"{e[k][j]:>16.7E}")
                    fh.write("\n")

# the text form is the reference; the binary is the same arrays written raw

DISP_FMT = {
    # the two waves were always written with different field widths
    "rayleigh": "{:.0f} " + "{:12.6f} "*3 + "{:12.4f} " + "{:16E} " + "{} "*2,
    "love": "{}" + "{:12.6f} " + "{:12.6f} "*2 + "{:15.4f} " + "{:15.7E} "
            + "{:5} "*2,
}

EIG_HDR_FMT = "{:3}" + 3*"{:13.7f} " + "{:15.7f}    {:4}    {:2d}"

NEIG_COL = {"rayleigh": 4, "love": 2}

FILE_TAG_MODE = {"disp": b"PSDISP01", "eigen": b"PSEIGN01"}

WAVE_CODE = {"rayleigh": 1, "love": 2}

WAVE_NAME = {1: "rayleigh", 2: "love"}

_DISP_I32 = ("mode", "ls", "ipd")

_DISP_F64 = ("period", "c", "u", "q", "flan")

_EIG_I32 = ("mode", "ls", "intper")

_EIG_F64 = ("period", "c", "u", "calc")

def parse_disp(text, wave):
    """The dispersion file as arrays. Inverse of format_disp."""
    lines = text.splitlines()

    # header: the model, then the same model after flattening
    nparam = int(lines[0].split()[0])
    nlayer = int(lines[1].split()[0])
    model = np.array([[float(x) for x in lines[2 + i].split()]
                      for i in range(nlayer)])
    i = 2 + nlayer
    assert lines[i].strip() == "flattened model", lines[i]
    model_flat = np.array([[float(x) for x in lines[i + 1 + k].split()]
                           for k in range(nlayer)])
    i += 1 + nlayer
    nbranch = int(lines[i].split()[0])
    i += 1

    # one line per (branch, period), optionally + 3 partial-derivative blocks
    mode, per, c, u, q, flan, ls, ipd, done = [], [], [], [], [], [], [], [], []
    der, der_off = [], [0]
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("*"):          # "**** mode number: N done ****"
            done.append(len(mode))
            i += 1
            continue
        p = ln.split()
        mode.append(int(p[0])); per.append(float(p[1])); c.append(float(p[2]))
        u.append(float(p[3])); q.append(float(p[4])); flan.append(float(p[5]))
        ls.append(int(p[6])); ipd.append(int(p[7]))
        i += 1
        if ipd[-1]:
            want, got = 3*ls[-1], []
            while len(got) < want:
                got += [float(x) for x in lines[i].split()]
                i += 1
            der += got
        der_off.append(len(der))

    return dict(kind="disp", wave=wave, nparam=nparam, nlayer=nlayer,
                model=model, model_flat=model_flat, nbranch=nbranch,
                mode=np.array(mode), period=np.array(per), c=np.array(c),
                u=np.array(u), q=np.array(q), flan=np.array(flan),
                ls=np.array(ls), ipd=np.array(ipd),
                done_at=np.array(done, dtype=np.int64),
                der=np.array(der), der_offset=np.array(der_off, dtype=np.int64))

def format_disp(d):
    """Arrays back to text, byte for byte."""
    out = [f"{int(d['nparam'])} no. of model parameters for part. der.",
           f"{int(d['nlayer'])} no. of layers in model"]
    out += [MODEL_FMT.format(*row) for row in d["model"]]
    out.append(" flattened model ")
    out += [MODEL_FMT.format(*row) for row in d["model_flat"]]
    out.append("{:3} modes listed in file".format(int(d["nbranch"])))

    fmt = DISP_FMT[str(d["wave"])]
    done = set(int(x) for x in np.atleast_1d(d["done_at"]))
    ndone = 0
    for k in range(len(d["mode"])):
        while k in done and ndone <= k:
            out.append(" ******** mode number: {:3d} done. ********"
                       .format(int(d["mode"][k - 1]) if k else 0))
            done.discard(k)
            ndone += 1
        out.append(fmt.format(int(d["mode"][k]), float(d["period"][k]),
                              float(d["c"][k]), float(d["u"][k]),
                              float(d["q"][k]), float(d["flan"][k]),
                              int(d["ls"][k]), int(d["ipd"][k])))
        if d["ipd"][k]:
            blk = d["der"][d["der_offset"][k]:d["der_offset"][k + 1]]
            ls = int(d["ls"][k])
            for j in range(3):
                vals, line = blk[j*ls:(j + 1)*ls], ""
                for i, v in enumerate(vals, start=1):
                    line += "{:.7E}".format(v) + " "
                    if i % 7 == 0 and i != ls:
                        out.append(line); line = ""
                out.append(line)
    for _ in range(len(done)):
        out.append(" ******** mode number: {:3d} done. ********"
                   .format(int(d["mode"][-1])))
    return "\n".join(out) + "\n"

def parse_eigen(text, wave):
    """The eigenfunction file as arrays. Inverse of format_eigen."""
    lines = text.splitlines()
    banner = lines[0]
    nlayer = int(lines[1].split()[0])
    model = np.array([[float(x) for x in lines[2 + i].split()]
                      for i in range(nlayer)])
    i = 2 + nlayer
    nbranch = int(lines[i].split()[0])
    i += 1

    # one block per branch, then one record per period inside it
    ncol = NEIG_COL[wave] + 1                    # depth + the eigenfunctions
    branch, per_mint, per_mall = [], [], []
    brk, mode, per, c, u, calc, ls, intper = [], [], [], [], [], [], [], []
    eig, eig_off = [], [0]
    while i < len(lines):
        if lines[i].rstrip().endswith("mode number"):
            branch.append(int(lines[i].split()[0]))
            p = lines[i + 1].split()
            per_mint.append(int(p[0])); per_mall.append(int(p[1]))
            brk.append(len(mode))
            i += 2
            continue
        p = lines[i].split()
        mode.append(int(p[0])); per.append(float(p[1])); c.append(float(p[2]))
        u.append(float(p[3])); calc.append(float(p[4]))
        ls.append(int(p[5])); intper.append(int(p[6]))
        i += 1
        for _ in range(ls[-1]):
            eig += [float(x) for x in lines[i].split()]
            i += 1
        eig_off.append(len(eig))

    return dict(kind="eigen", wave=wave, banner=banner, nlayer=nlayer,
                model=model, nbranch=nbranch, ncol=ncol,
                branch=np.array(branch), per_mint=np.array(per_mint),
                per_mall=np.array(per_mall), branch_start=np.array(brk),
                mode=np.array(mode), period=np.array(per), c=np.array(c),
                u=np.array(u), calc=np.array(calc), ls=np.array(ls),
                intper=np.array(intper), eig=np.array(eig),
                eig_offset=np.array(eig_off, dtype=np.int64))

def format_eigen(d):
    """Arrays back to text, byte for byte."""
    out = [str(d["banner"]), f"{int(d['nlayer'])} no. of layers in model"]
    out += [MODEL_FMT.format(*row) for row in d["model"]]
    out.append("{:3} modes listed in file".format(int(d["nbranch"])))

    wave = str(d["wave"])
    ncol = int(d["ncol"])
    starts = {int(v): k for k, v in enumerate(d["branch_start"])}
    for k in range(len(d["mode"])):
        if k in starts:
            b = starts[k]
            out.append("{:3}".format(int(d["branch"][b])) + " mode number")
            out.append("{:3} {:4}      period samples this mode (integral, all)"
                       .format(int(d["per_mint"][b]), int(d["per_mall"][b])))
        out.append(EIG_HDR_FMT.format(int(d["mode"][k]), float(d["period"][k]),
                                      float(d["c"][k]), float(d["u"][k]),
                                      float(d["calc"][k]), int(d["ls"][k]),
                                      int(d["intper"][k])))
        blk = d["eig"][d["eig_offset"][k]:d["eig_offset"][k + 1]]
        for r in range(int(d["ls"][k])):
            row = blk[r*ncol:(r + 1)*ncol]
            if wave == "rayleigh":
                out.append("".join("{:10.6E} ".format(v) for v in row))
            else:
                out.append("".join("{:15.7E}".format(v) for v in row))
    return "\n".join(out) + "\n"

def write_modefile(d, path):
    """Write the binary form of a parsed disp or eigen file."""
    kind = str(d["kind"])
    with open(path, "wb") as fh:
        fh.write(FILE_TAG_MODE[kind])
        if kind == "disp":
            fh.write(struct.pack("<6i", WAVE_CODE[str(d["wave"])],
                                 int(d["nparam"]), int(d["nlayer"]),
                                 int(d["nbranch"]), len(d["mode"]),
                                 len(d["done_at"])))
            fh.write(struct.pack("<q", len(d["der"])))
            _put(fh, d["model"], "<f8")
            _put(fh, d["model_flat"], "<f8")
            _put(fh, d["done_at"], "<i4")
            for k in _DISP_I32:
                _put(fh, d[k], "<i4")
            for k in _DISP_F64:
                _put(fh, d[k], "<f8")
            _put(fh, d["der_offset"], "<i8")
            _put(fh, d["der"], "<f8")
        else:
            banner = str(d["banner"]).encode()
            fh.write(struct.pack("<6i", WAVE_CODE[str(d["wave"])],
                                 int(d["nlayer"]), int(d["nbranch"]),
                                 len(d["mode"]), len(d["branch"]),
                                 int(d["ncol"])))
            fh.write(struct.pack("<q", len(d["eig"])))
            fh.write(struct.pack("<i", len(banner)))
            fh.write(banner)
            _put(fh, d["model"], "<f8")
            for k in ("branch", "per_mint", "per_mall", "branch_start"):
                _put(fh, d[k], "<i4")
            for k in _EIG_I32:
                _put(fh, d[k], "<i4")
            for k in _EIG_F64:
                _put(fh, d[k], "<f8")
            _put(fh, d["eig_offset"], "<i8")
            _put(fh, d["eig"], "<f8")

def read_modefile(path):
    """Read the binary form back into the dict parse_disp/parse_eigen makes."""
    with open(path, "rb") as fh:
        tag = fh.read(8)
        if tag == FILE_TAG_MODE["disp"]:
            wave, nparam, nlayer, nbranch, nrec, ndone = \
                struct.unpack("<6i", fh.read(24))
            nder, = struct.unpack("<q", fh.read(8))
            d = dict(kind="disp", wave=WAVE_NAME[wave], nparam=nparam,
                     nlayer=nlayer, nbranch=nbranch)
            d["model"] = _get(fh, nlayer*4, "<f8").reshape(nlayer, 4)
            d["model_flat"] = _get(fh, nlayer*4, "<f8").reshape(nlayer, 4)
            d["done_at"] = _get(fh, ndone, "<i4").astype(np.int64)
            for k in _DISP_I32:
                d[k] = _get(fh, nrec, "<i4")
            for k in _DISP_F64:
                d[k] = _get(fh, nrec, "<f8")
            d["der_offset"] = _get(fh, nrec + 1, "<i8")
            d["der"] = _get(fh, nder, "<f8")
            return d
        if tag != FILE_TAG_MODE["eigen"]:
            raise ValueError(f"{path}: not a SW1D mode file")
        wave, nlayer, nbranch, nrec, nbr, ncol = struct.unpack("<6i", fh.read(24))
        neig, = struct.unpack("<q", fh.read(8))
        nb, = struct.unpack("<i", fh.read(4))
        d = dict(kind="eigen", wave=WAVE_NAME[wave],
                 banner=fh.read(nb).decode(), nlayer=nlayer,
                 nbranch=nbranch, ncol=ncol)
        d["model"] = _get(fh, nlayer*4, "<f8").reshape(nlayer, 4)
        for k in ("branch", "per_mint", "per_mall", "branch_start"):
            d[k] = _get(fh, nbr, "<i4")
        for k in _EIG_I32:
            d[k] = _get(fh, nrec, "<i4")
        for k in _EIG_F64:
            d[k] = _get(fh, nrec, "<f8")
        d["eig_offset"] = _get(fh, nrec + 1, "<i8")
        d["eig"] = _get(fh, neig, "<f8")
        return d

def format_modefile(d):
    """Either mode file, from arrays back to its text layout."""
    return format_disp(d) if str(d["kind"]) == "disp" else format_eigen(d)
