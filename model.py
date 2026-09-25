"""Layer arrays, interfaces at the source and receiver, flattening, Q, bounds."""
import math
import numpy as np

from common import common, dprint

def isInteger(x,tol=1e-5):
    """True when x is within tol of a whole number."""
    return abs(x-round(x)) < tol

def initialise(nelements):
    """Allocate the shared model and scratch arrays for nelements layers."""
    common.vps=np.zeros(nelements)
    common.vss=np.zeros(nelements)
    common.ro=np.zeros(nelements)
    common.qb=np.zeros(nelements)
    common.qa=np.zeros(nelements)
    common.d0=np.zeros(nelements)
    common.vp0=np.zeros(nelements)
    common.vs0=np.zeros(nelements)
    common.ro0=np.zeros(nelements)
    common.qb0=np.zeros(nelements)
    common.qa0=np.zeros(nelements)
    common.vs=np.zeros(nelements)
    common.vp=np.zeros(nelements)
    common.fu=np.zeros(nelements)
    common.idep=np.zeros(nelements)
    common.d=np.zeros(common.lyrs)
    common.ist=0
    common.der=np.zeros((3,common.lyrs))
    common.de=np.zeros(2)
    common.u=0
    common.um=0
    common.nord=0

def rsplit(rdep):
    """Insert an interface at the receiver depth, if one is not there already."""
    if(rdep <= 0):
        common.irdep=1
        return
    thk=0.0
    iss=0
    while thk < rdep and iss < common.n:
        thk+=common.d[iss]
        iss+=1
    common.irdep = iss
    if thk != rdep :  # Make new layer only if the receriver is not already at the interface
        splt = thk - rdep #depth of new layer
        for l in range(0,common.n - iss ):
            k = common.n -l -1
            j = k+1
            common.d[j] = common.d[k]
            common.ro[j] = common.ro[k]
            common.vps[j] = common.vps[k]
            common.vss[j] = common.vss[k]
            common.qa[j] = common.qa[k]
            common.qb[j] = common.qb[k]
        common.d[iss] = splt
        common.d[iss - 1] = common.d[iss - 1] -splt
        common.ro[iss] = common.ro[iss - 1]
        common.vps[iss] = common.vps[iss - 1]
        common.vss[iss] = common.vss[iss - 1]
        common.qa[iss] = common.qa[iss - 1]
        common.qb[iss] = common.qb[iss - 1]
        common.n+=1

def flat(jcom, iefl):

    """Earth-flattening transform, applied to the model in place."""
    dprint("===============")
    dprint("FUNCTION: flat")
    a=6371.0
    if(iefl==0):
        return
    pwr=2.275
    if(abs(jcom)>1):
        pwr=5.0
    atp=a**pwr
    nm=common.n-1
    hs=0.0
    dprint("common.d= ",common.d)
    for i in range(1,common.n+1):
        ht=hs
        hs=hs+common.d[i-1]
        common.d[i-1]=a-ht
    dprint("common.d= ",common.d)
    for i in range(1,nm+1):
        ii=i+1
        dprint("common.d[ii-1]: ",common.d[ii-1])
        dprint("common.d[i-1]: ",common.d[i-1])
        fltd=math.log(common.d[i-1]/common.d[ii-1])
        dif=((1.0/common.d[ii-1])-(1.0/common.d[i-1]))*a/fltd
        difr=(common.d[i-1]**pwr)-(common.d[ii-1]**pwr)
        dprint(fltd,dif,difr)
        common.ro[i-1]=common.ro[i-1]*difr/(fltd*(a**pwr)*pwr)
        common.vps[i-1]=common.vps[i-1]*dif
        common.vss[i-1]=common.vss[i-1]*dif

    n=common.n
    fact= a/common.d[n-1]
    facti=(common.d[n-1]**pwr)/atp
    common.vps[n-1]=common.vps[n-1]*fact
    common.vss[n-1]=common.vss[n-1]*fact
    common.ro[n-1]=common.ro[n-1]*facti
    z0=0.0
    dprint(common.vps[n-1],common.vss[n-1],common.ro[n-1])
    for i in range(2,n+1):
        z1=a*math.log(a/common.d[i-1])
        common.d[i-2]=z1-z0
        z0=z1

    common.d[n-1]=0.0
    dprint("common.d at the end: ", common.d)
    return

def split(rdep):
    """Insert an interface at each source depth, as rsplit does for the receiver."""
    for ndp in range(0,common.nsrce):
        thk = 0.0
        iss =0
        dprint(ndp)
        if(common.sdep[ndp] <= 0.0):
            common.idep[ndp] = iss+1        # layer index holding this source
            continue

        while thk < common.sdep[ndp] and iss < common.n:
            thk+=common.d[iss]
            iss+=1


        common.idep[ndp] = iss+1
        if thk != common.sdep[ndp] :  # Make new layer only if the receriver is not already at the interface
            splt = thk - common.sdep[ndp] #depth of new layer
            for l in range(0,common.n - iss ):
                k = common.n -l -1
                j = k+1
                common.d[j] = common.d[k]
                common.ro[j] = common.ro[k]
                common.vps[j] = common.vps[k]
                common.vss[j] = common.vss[k]
                common.qa[j] = common.qa[k]
                common.qb[j] = common.qb[k]
            common.d[iss] = splt
            common.d[iss - 1] = common.d[iss - 1] -splt
            common.ro[iss] = common.ro[iss - 1]
            common.vps[iss] = common.vps[iss - 1]
            common.vss[iss] = common.vss[iss - 1]
            common.qa[iss] = common.qa[iss - 1]
            common.qb[iss] = common.qb[iss - 1]
            common.n+=1
            if common.sdep[ndp] < rdep:
                common.irdep+=1

def qcor(om, omref):
    """Apply the causal Q dispersion correction at angular frequency om."""
    dprint("===============")
    dprint("FUNCTION: qcor")
    omscl = 0
    if(omref != 0):
        omscl = np.log(om/omref) / np.pi
    dprint("omscl= ",omscl)
    for i in range(0,common.n):
        common.vs[i] = common.vss[i]*((1.0 + common.qb[i]*omscl)**2)
        common.vp[i] = common.vps[i]*((1.0 + common.qa[i]*omscl)**2)
        common.fu[i] = common.ro[i] * common.vs[i]

def cmaximum(c2):
    """Phase-velocity search ceiling: the half-space shear velocity, or c2."""
    dprint("===============")
    dprint("FUNCTION: cmaximum")
    cmax = math.sqrt(common.vs[common.n - 1]) - 0.00000001
    cmx = 0
    if(c2 <= 0 ):
        cmx = cmax
    else:
        cmx = min(cmax , c2)
    dprint("cmx= ",cmx)
    return cmx
