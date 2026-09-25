"""Sweeping the frequency grid, branch by branch."""
import copy
import io
import math
import numpy as np
import struct

from common import common, dprint
from model import cmaximum, flat, initialise, isInteger, qcor, rsplit, split
from modecount import detk
from search import cex, intrp
from modes import Collector
from modefiles import after_flat, before_flat, nbran_line

def compute(nb):
    """"""
    dprint("===============")
    dprint("FUNCTION: compute")
    common.ctry = 0.5*(common.cmn + common.cmx)
    common.ceps = 0.5*(common.cmx - common.ctry)
    common.cm = 0

    real_iouf1 = common.iouf1
    common.iouf1 = io.BytesIO()
    if(common.iasc):
        real_s10 = common.s10
        common.s10 = io.StringIO()

    per_mall = 0
    per_mint = 0
    iusr = 1
    for i in range(1,common.nh+1):
        om = common.omax - i*common.dom
        if(common.nusrper>0 and iusr<=common.nusrper):
            if(common.usrom[iusr-1]>om):
                dprint("Found gap at", 2*np.pi/(om), "for", 2*np.pi/common.usrom[iusr-1],'\n')
                if(common.omuse!=common.usrom[iusr-1]):
                    common.omuse = common.usrom[iusr-1]
                else:
                    common.omuse=om
                iusr=iusr+1
            else:
                common.omuse = om
        else:
            common.omuse = om

        qcor(common.omuse,common.omref)
        common.cmx = cmaximum(common.c2)

        nev = cex(common.omuse,nb,common.jcom)
        dprint(" Second loop: nev= ",nev)
        if(not nev):
            break
        common.wrote_record = True
        intrp(common.omuse,common.dom,common.jcom)
        if(common.wrote_record):
            per_mall += 1
            if(isInteger(2*np.pi/common.omuse)):
                per_mint += 1

    buffered_bin = common.iouf1.getvalue()
    common.iouf1 = real_iouf1
    if(common.iasc):
        buffered_asc = common.s10.getvalue()
        common.s10 = real_s10
        common.s10.write('{:3}'.format(nb)+" mode number\n")
        common.s10.write("{:3} {:4}      period samples this mode (integral, all)\n".format(per_mint, per_mall))
        common.s10.write(buffered_asc)
        common.iouf2.write(" ******** mode number: {:3d} done. ********".format(nb)+'\n')

    common.iouf1.write(struct.pack('i',nb))
    common.iouf1.write(struct.pack('i', per_mall))
    common.iouf1.write(buffered_bin)
    # istop marks the last branch written to this file
    common.iouf1.write(struct.pack('i', 1 if nb==common.nbran2 else 0))

def run(cfg, collect=None):
    """Solve one wave type. Appends modes to `collect`; returns the file text."""

    common.COLLECT = collect
    common.iasc = bool(getattr(cfg, "mode_files", False))
    # built in memory; the caller decides what to write, so nothing opens here
    common.iouf2 = io.StringIO() if common.iasc else None
    common.s10 = io.StringIO() if common.iasc else None
    common.iouf1 = io.BytesIO()

    common.n0 = len(cfg.thickness)
    common.iefl = 1 if cfg.flatten else 0
    common.tref = cfg.tref
    common.omref = 0.0
    initialise(common.lyrs)
    if common.tref != 0:
        common.omref = 2*np.pi/common.tref

    m = cfg
    for i in range(common.n0):
        common.d0[i]  = m.thickness[i]
        common.vp0[i] = m.vp[i]
        common.vs0[i] = m.vs[i]
        common.ro0[i] = m.rho[i]
        common.qb0[i] = 1.0/m.qbeta[i]  if m.qbeta[i]  != 0 else 0.0
        common.qa0[i] = 1.0/m.qalpha[i] if m.qalpha[i] != 0 else 0.0

    common.jcom = 1 if cfg.wave == "rayleigh" else 2
    common.usrper = np.array(sorted(cfg.ensure_periods), dtype=float)
    common.nusrper = len(common.usrper)
    common.usrom = 2*np.pi/common.usrper if common.nusrper else np.array([])

    common.c1, common.c2 = cfg.cmin, cfg.cmax
    common.nbran1, common.nbran2 = cfg.mode_min, cfg.mode_max
    common.nsrce = len(cfg.source_depths)
    common.nom = cfg.nfreq
    common.df = cfg.df
    common.fo = cfg.f0
    common.n = common.n0

    for i in range(common.n):
        common.d[i]   = common.d0[i]
        common.ro[i]  = common.ro0[i]
        common.vps[i] = common.vp0[i]
        common.vss[i] = common.vs0[i]
        common.qa[i]  = common.qa0[i]
        common.qb[i]  = common.qb0[i]

    common.sdep = sorted(cfg.source_depths) + [0.0]*(common.lsd - common.nsrce)
    common.rdep = cfg.receiver_depth
    rsplit(common.rdep)
    split(common.rdep)

    if common.iasc:
        before_flat(common.iouf2, common.s10, common)

    flat(common.jcom, common.iefl)

    if common.iasc:
        after_flat(common.iouf2, common)

    common.noc = 1
    if common.vss[0] <= 0:
        common.noc = 2
    cmin_ = min(0.60*common.vss[common.noc-1], common.vps[0])
    common.cmn = max(common.c1, cmin_)

    common.vps = np.square(common.vps)
    common.vss = np.square(common.vss)

    common.dom = 2*np.pi*common.df
    common.nh = common.nom
    common.omax = 2*np.pi*common.fo + common.nh*common.dom
    common.om = common.omax - common.dom
    qcor(common.om, common.omref)
    common.cmax = math.sqrt(common.vs[common.n-1]) - 1e-8
    common.cmx = cmaximum(common.c2)

    common.kei, common.dei = detk(common.cmx, common.om, common.jcom)
    if common.nbran2 < 0 or common.nbran2 >= common.kei:
        common.nbran2 = common.kei - 1
    common.nbran = max(common.nbran2 - common.nbran1 + 1, 1)

    if common.iasc:
        nbran_line(common.iouf2, common.s10, common.nbran)

    if collect is not None:
        collect.begin(cfg, common.jcom, common.nbran)

    common.iouf1.write(struct.pack('iiddii', common.nsrce, common.nom, common.df, common.fo, common.jcom, common.nbran))
    for i in range(common.nsrce):
        common.iouf1.write(struct.pack('d', common.sdep[i]))

    for nb in range(common.nbran1, common.nbran2+1):
        if collect is not None:
            collect.begin_branch(nb)
        compute(nb)

    if collect is not None:
        collect.end()
    return (common.iouf1.getvalue(),
            common.iouf2.getvalue() if common.iasc else None,
            common.s10.getvalue() if common.iasc else None)

def solve(cfg, wave, verbose=False, eigenfunctions=False, mode_files=False):
    """Solve one wave type. Returns (Modes, disp text, eigen text)."""
    if wave not in ("rayleigh", "love"):
        raise ValueError("wave must be 'rayleigh' or 'love'")
    c = copy.copy(cfg)
    c.wave = wave
    c.mode_files = bool(mode_files)

    common.VERBOSE = bool(verbose)
    col = Collector(want_eigen=eigenfunctions)
    _, disp_text, eigen_text = run(c, collect=col)
    return col.to_modes(), disp_text, eigen_text

def solve_both(cfg, verbose=False, eigenfunctions=False):
    """Returns (rayleigh Modes, love Modes)."""
    ray = solve(cfg, "rayleigh", verbose, eigenfunctions)[0]
    lov = solve(cfg, "love", verbose, eigenfunctions)[0]
    return ray, lov
