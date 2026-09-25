"""Shared solver state: the model, the scratch arrays and the counters."""
import numpy as np

class common:
    """Model, arrays and counters shared by the whole solver."""

    lyrs=3000
    lsd=4
    lgrm=2050
    lgrm2=lgrm*2

    maxrecs=8
    maxpee=10

    maxcol=lyrs*3
    maxrow=maxrecs*lgrm+maxcol+maxpee
    maxsect=maxcol*lgrm

    maxbrn=50
    maxbrn2=maxbrn*2

    VERBOSE = False

    d0 = []
    vp0 = []
    vs0 = []
    ro0 = []
    qb0 = []
    qa0 = []
    n0=0

    d = []
    vps = []
    vss = []
    ro = []
    qb = []
    qa = []
    n=0

    sdep=[0]*lsd

    irdep=0

    wrote_record = True

def dprint(*args, **kwargs):
    """print(), gated on common.VERBOSE, for tracing a single solve."""
    if common.VERBOSE:
        print(*args, **kwargs)

# xknt, yknt are rayknt/propagate's minor vectors

common.ctr = 0

common.yknt = np.zeros(5)

common.xknt = np.zeros(5)

common.coef=0

common.ce=np.zeros(2)

common.de=np.zeros(2)

common.ke=np.zeros(2)

common.b=np.zeros([2,4])

common.COLLECT = None   # a Collector while run() is executing

common.dep=np.zeros(common.lyrs)

common.depth = np.zeros(common.lyrs)

common.per_mall = 0

common.intper = 0

common.permint = 0

common.nusrper = 0

common.outfil = ""

common.outfil_exasc = ""

common.ym = np.zeros((5,common.lyrs))

common.scale = np.zeros(common.lyrs)

common.y=np.zeros(5)

common.x=np.zeros((4,common.lyrs))

common.omuse=0

common.deriv_count=0
