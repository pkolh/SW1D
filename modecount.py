"""Sturm count: how many branches lie below a trial phase velocity."""
import numpy as np
from math import sin, cos, sqrt, atan

from common import common, dprint
from propagator import arg2, propagate, strtdp

def lovknt(cc, w):
    """Count the Love branches with phase velocity below cc at frequency w."""
    dprint("===============")
    dprint("FUNCTION: lovknt")
    kount = 0
    p = 1.0 / cc
    psq = p*p

    strtdp(psq, w)

    y1 = 1.0
    y2 = -1 * common.fu[common.ist-1] * sqrt(psq -
            1.0/common.vs[common.ist-1]) # fu = ro * vs
    i = common.ist
    pi = np.pi
    d=common.d
    fu=common.fu
    vs=common.vs
    noc = common.noc

    def foo2(w,kount,psq,y1,y2,i,pi,d,fu,vs,noc):
        """Propagate the Love 2-vector upward, counting sign changes."""
        dprint("===============")
        dprint("FUNCTION: foo2")
        while True:
            i=i-1
            hb = psq - 1/common.vs[i-1]
            if( hb < 0):
                hh = sqrt(-hb)
                wh = w*hh
                a1 = common.fu[i-1]*hb*y1
                c1 = 0.5 * pi/wh
                if(a1 != 0):
                    c1 = atan(y2*hh/a1)/wh
                pih = pi/wh
                k=-1
                while True:
                    k=k+1
                    xtry = c1 + k*pih
                    if(xtry<=common.d[i-1]):
                        if(xtry<=0):
                            continue
                        else:
                            kount=kount+1
                            dprint("kount= ",kount)
                    else:
                        break
                th=wh*common.d[i-1]
                cb=cos(th)
                sb=-sin(th)/hh
                x1=cb*y1+sb*y2/common.fu[i-1]
                x2=hb*sb*common.fu[i-1]*y1 +cb*y2
                y1=x1
                y2=x2
                if(i>common.noc):
                    continue
                else:
                    de=y2/(abs(y2)+abs(y1))
                    return(kount,y1,y2)
            else:
                wd=w*common.d[i-1]
                cb,sb = arg2(wd, hb)
                x1 = cb*y1 + sb*y2/common.fu[i-1]
                x2 = hb*sb*common.fu[i-1]*y1 + cb*y2
                if(x2*y2 < 0 and hb>=0):
                    kount=kount-1
                    dprint("kount= ",kount)
                y1 = x1
                y2 = x2
                if(i>common.noc):
                    continue
                else:
                    break
        return kount,y1,y2
    kount,y1,y2 = foo2(w,kount,psq,y1,y2,i,pi,d,fu,vs,noc)
    dprint("kount= ",kount)
    de = y2/(abs(y2) + abs(y1))
    dprint("de= ",de)
    return (kount, de)

def rayknt(cc, w):
    """Count the Rayleigh branches with phase velocity below cc at frequency w."""
    dprint("===============")
    dprint("FUNCTION: rayknt")
    kount = 0
    p = 1.0 / cc
    psq = p*p
    strtdp(psq , w)
    r2 = 2 * common.fu[common.ist-1] * p
    dprint("r2= ",r2)
    common.yknt[2] = sqrt(psq - 1/common.vp[common.ist-1])
    common.yknt[3] = -sqrt(psq - 1/common.vs[common.ist-1])
    common.yknt[0] = -(common.yknt[2]*common.yknt[3] + psq)/common.ro[common.ist-1]
    common.yknt[1] = r2*common.yknt[0] + p
    common.yknt[4] = common.ro[common.ist-1] - r2*(p+common.yknt[1])
    dprint("y[0]= ",common.yknt[0])
    dprint("y[1]= ",common.yknt[1])
    dprint("y[2]= ",common.yknt[2])
    dprint("y[3]= ",common.yknt[3])
    dprint("y[4]= ",common.yknt[4])
    if(common.yknt[4] > 0 ):
        kount = 1
    dprint("kount= ",kount)
    i = common.ist

    return(propagate(i,p,kount,psq,w))

def detk(cc, w, jcom):
    """Mode count at trial phase velocity cc, for whichever wave jcom names."""
    dprint("===============")
    dprint("FUNCTION: detk")
    if (jcom == 1):
        return(rayknt(cc,w))
    elif ( jcom == 2 ):
        return lovknt(cc,w)
