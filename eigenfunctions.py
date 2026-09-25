"""Rebuilding and rescaling the eigenfunction once a root has converged."""
import numpy as np
from math import exp

from common import common, dprint
from propagator import arg, solve_coef

def fixlov(w,p,x):
    """Rebuild the Love eigenfunction at the converged root, rescaled per layer."""
    dprint("===============")
    dprint("FUNCTION: fixlov")
    psq = p*p
    lvz = common.ist + 1
    for i in range(common.noc, common.ist+1):
        lvz -= 1
        hb = psq - 1.0/common.vs[lvz-1]
        if(hb<0):
            break
    dprint(" **fixlov entered-redo propagation to layer {} :",lvz)
    lvz=max(common.noc,lvz-1)
    dprint("lvz= ",lvz)
    x1s=x[0][lvz-1]
    x2s=x[1][lvz-1]
    x[0][common.noc-1]=1.0
    x[1][common.noc-1]=0.0
    i=common.noc
    dprint("Pre-Loop")
    while(True):
        ii=i+1
        hb=psq-1.0/common.vs[i-1]
        wd=w*common.d[i-1]
        cb,sb,facb=arg(wd,hb)
        dfac=exp(facb)
        sb=sb*dfac
        cb=cb*dfac
        x[0][ii-1]=cb*x[0][i-1]-sb*x[1][i-1]/common.fu[i-1]
        x[1][ii-1]=cb*x[1][i-1]-hb*sb*common.fu[i-1]*x[0][i-1]
        i=ii
        if(i<lvz):
            continue
        else:
            break
    dprint("Post-loop")
    scl=(x[0][lvz-1]*x1s+x[1][lvz-1]*x2s)/(x1s*x1s+x2s*x2s)
    lvz1=lvz+1
    for i in range(lvz1,common.ist+1):
        for j in range(1,3):
            x[j-1][i-1]=scl*x[j-1][i-1]
    return x

def fixray(ysav,w,p,x):
    """Rebuild the Rayleigh eigenfunction from the converged minor vector."""
    dprint("===============")
    dprint("FUNCTION: fixray")
    coef=0
    xx=np.zeros(4)
    psq=p*p
    lvz=common.ist+1
    for i in range(common.noc,common.ist+1):
        lvz=lvz-1
        hb=psq-1/common.vs[lvz-1]
        if(hb<0):
            break
    dprint("**fixray entered-redo propagation to layer ",lvz)
    dprint("TEST 0")
    lvz=max(common.noc,lvz-1)
    dprint("lvz= ",lvz)
    dprint("shape= ",np.shape(x))
    for i in range(1,5):
        xx[i-1]=x[i-1][lvz-1]
    dprint("initial xx= ",xx)
    dprint("initial x= ",x)
    dprint("TEST 1")
    for i in range(1,3):
        for j in range(1,5):
            common.b[i-1][j-1]=0.0
    common.b[0][0]=1.0
    common.b[1][1]=1.0
    common.b[0][2]=ysav
    cosav=0.0
    i=common.noc
    dprint("fixray pre-loop")
    while(True):
        ii=i+1
        r2=2.0*common.fu[i-1]*p
        wd=w*common.d[i-1]
        ha=psq-1.0/common.vp[i-1]
        hb=psq-1.0/common.vs[i-1]
        ca,sa,faca=arg(wd,ha)
        cb,sb,facb=arg(wd,hb)
        dfac1=exp(faca)
        sa=sa*dfac1
        ca=ca*dfac1
        dfac2=exp(facb)
        sb=sb*dfac2
        cb=cb*dfac2
        dprint("Inner Loop Begin")
        for k in range(1,3):
            e2=r2*common.b[k-1][1]-common.b[k-1][2]
            e3=common.ro[i-1]*common.b[k-1][1]-p*e2
            e4=r2*common.b[k-1][0]-common.b[k-1][3]
            e1=common.ro[i-1]*common.b[k-1][0]-p*e4
            e6=ca*e2-sa*e1
            e8=cb*e4-sb*e3
            common.b[k-1][0]=(ca*e1-ha*sa*e2+p*e8)/common.ro[i-1]
            common.b[k-1][1]=(cb*e3-hb*sb*e4+p*e6)/common.ro[i-1]
            common.b[k-1][2]=r2*common.b[k-1][1]-e6
            common.b[k-1][3]=r2*common.b[k-1][0]-e8
        dprint("Inner Loop Over")
        dprint("coef= ",coef)
        coef=solve_coef(ii,x,common.b)
        dprint("coef= ",coef)
        dco=abs((coef-cosav)/coef)
        cosav=coef
        i=ii
        if(i==lvz):
            break
        else:
            if(dco>1e-9):
                 continue
            else:
                  break
    dprint("fixray postloop 1")
    dprint("coef= ",coef)
    dprint("ysav= ",ysav)
    x[0][common.noc-1]=1.0
    x[1][common.noc-1]=coef
    x[2][common.noc-1]=ysav
    x[3][common.noc-1]=0.0
    i=common.noc
    depth=0
    while(True):

        ii=i+1
        dprint("ii= ",ii)
        r2=2.0*common.fu[i-1]*p
        wd=w*common.d[i-1]

        #for testing:
        depth=depth+common.d[i-1]
        dprint("depth= ",depth)
        if(depth==185):
            dprint("PROBLEM DEPTH")
        ha=psq-1.0/common.vp[i-1]
        hb=psq-1.0/common.vs[i-1]
        ca,sa,faca=arg(wd,ha)
        cb,sb,facb=arg(wd,hb)
        dfac1=exp(faca)
        sa=sa*dfac1
        ca=ca*dfac1
        dfac2=exp(facb)
        sb=sb*dfac2
        cb=cb*dfac2
        dprint("x[0][i-1]= ",x[0][i-1])
        dprint("x[1][i-1]= ",x[1][i-1])
        dprint("x[2][i-1]= ",x[2][i-1])
        dprint("x[3][i-1]= ",x[3][i-1])
        e2=r2*x[1][i-1]-x[2][i-1]
        e3=common.ro[i-1]*x[1][i-1]-p*e2
        e4=r2*x[0][i-1]-x[3][i-1]
        e1=common.ro[i-1]*x[0][i-1]-p*e4
        e6=ca*e2-sa*e1
        e8=cb*e4-sb*e3
        dprint(e1,e2,e3,e4,e6,e8,sb,cb,sa,ca,ha,hb,r2,wd)
        dprint("common.ro[i-1]= ",common.ro[i-1])
        x[0][ii-1]=(ca*e1-ha*sa*e2+p*e8)/common.ro[i-1]
        x[0][ii-1]=round(x[0][ii-1],9)
        dprint("x[0][ii-1]= ",x[0][ii-1])
        x[1][ii-1]=(cb*e3-hb*sb*e4+p*e6)/common.ro[i-1]
        x[1][ii-1]=round(x[1][ii-1],9)
        dprint("x[1][ii-1]= ",x[1][ii-1])
        x[2][ii-1]=r2*x[1][ii-1]-e6
        x[2][ii-1]=round(x[2][ii-1],9)
        dprint("x[2][ii-1]= ",x[2][ii-1])
        x[3][ii-1]=r2*x[0][ii-1]-e8
        x[3][ii-1]=round(x[3][ii-1],9)
        dprint("x[3][ii-1]= ",x[3][ii-1])
        i=ii
        if(i<lvz):
            continue
        else:
            break
    dprint("fixray post-loop 2")
    xtxx=0.0
    xtx=0.0
    dprint("final xx= ",xx)
    for j in range(1,5):
        xtxx=xtxx+x[j-1][lvz-1]*xx[j-1]
        dprint("xtxx= ",xtxx)
        dprint("x[j-1][lvz-1]= ",x[j-1][lvz-1])
        xtx=xtx+xx[j-1]*xx[j-1]
        dprint("xtx= ",xtx)

    dprint("xtxx= ",xtxx)
    dprint("xtx= ",xtx)
    scl=xtxx/xtx
    dprint("scl= ",scl)
    lvz1=lvz+1

    for i in range(lvz1,common.ist+1):
        for j in range(1,5):
            x[j-1][i-1]=scl*x[j-1][i-1]

    return(x)
