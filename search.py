"""Finding a root: bracket a branch's phase velocity, then refine it."""
import math

from common import common, dprint
from propagator import strtdp
from modecount import detk
from dispersion import detray


def cex(om, nb, jcom):
    """Bracket branch nb's phase velocity at frequency om."""
    dprint("===============")
    dprint("FUNCTION: cex")
    common.tol = 2.e-9
    cepm = 20.0 * common.ceps
    nup = nb + 1
    cx = common.ctry
    common.ce[0] = 0.50 * common.cmn #ce[0] and ce[1] are kind of like minima(related to nb) and maxima(related to nb+1)
    common.ce[1] = 2 * common.cmx
    ke0 = 10000
    ke1 = -10000
    dprint("ke0= ",ke0)
    dprint("ke1= ",ke1)
    kx = 0
    ce0 = common.ce[0]
    ce1 = common.ce[1]
    de0 = common.de[0]
    de1 = common.de[1]
    cmn = common.cmn
    cmx = common.cmx
    ceps = common.ceps
    dprint("BEFORE foo")
    dprint("ce0= ",ce0)
    dprint("ce1= ",ce1)
    dprint("de0= ",de0)
    dprint("de1= ",de1)
    dprint("cmn= ",cmn)
    dprint("cmx= ",cmx)
    dprint("ceps= ",ceps)


    def foo(ke0,ke1,kx,ce0,ce1,cx,cmn,cmx,de0,de1,ceps):
        """Bisect on the mode count until the bracket holds exactly one branch."""
        dprint("===============")
        dprint("FUNCTION: foo")
        while (ke1 - ke0 != 1):
            cx = min(max(cx,cmn),cmx)
            dprint("cx= ",cx)
            kx, dx = detk(cx, om, jcom)
            dprint("kx,dx= ",kx,dx)
            dprint("Within Loop: detk called: kx,dx= ",kx,dx)
            if kx >= nup:
                dprint("IF clause: ")
                if cx <= cmn:
                    return (ce0,ce1,de0,de1,ceps,ke0,ke1,False)
                if(kx == nup and ke1 == nup): #Reached a "stationary state" i.e new kx didnt change
                    delc = 2.0*dx*(cx-ce1)/(dx-de1)
                    ceps = max(min(cepm,delc),ceps) #Strictly Increase ceps to delc(upto cepm)
                ce1 = cx
                ke1 = kx
                de1 = dx
                dprint("ce1,ke1,de1= ",ce1,ke1,de1)
                cx = cx - ceps
                if(cx <= ce0):
                    cx=(ce0+ce1)/2
            else:
                dprint("cx= ",cx)
                dprint("cmx= ",cmx)
                dprint("ELSE clause: ")
                if(cx >= cmx):
                    return (ce0,ce1,de0,de1,ceps,ke0,ke1,False)
                if(kx == nb and ke0 == nb): #Reached a "stationary state" i.e new kx didnt change
                    delc = -2.0*dx*(cx-ce0)/(dx-de0)
                    ceps = max(min(cepm,delc),ceps) #Strictly Increase ceps to delc(upto cepm)
                ce0 = cx
                ke0 = kx
                de0 = dx
                dprint("ce0,ke0,de0= ",ce0,ke0,de0)
                cx = cx + ceps
                if(cx >= ce1):
                    cx=(ce0+ce1)/2
        return (ce0,ce1,de0,de1,ceps,ke0,ke1,True)
    ce0,ce1,de0,de1,ceps,ke0,ke1,rev=foo(ke0,ke1,kx,ce0,ce1,cx,cmn,cmx,de0,de1,ceps)
    common.ce[0] = ce0
    common.ce[1] = ce1
    common.de[0] = de0
    common.de[1] = de1
    common.ceps = ceps
    common.ke[0]=ke0
    common.ke[1]=ke1
    dprint("AFTER foo")
    dprint("common.ce[0]= ",common.ce[0])
    dprint("common.ce[1]= ",common.ce[1])
    dprint("common.de[0]= ",common.de[0])
    dprint("common.de[1]= ",common.de[1])
    dprint("common.ke[0]= ",common.ke[0])
    dprint("common.ke[1]= ",common.ke[1])
    dprint("common.ceps= ",common.ceps)
    dprint("rev= ",rev)
    return rev

def intrp(om,dom,jcom):
    """Refine a bracketed root of the dispersion determinant."""
    dprint("===============")
    dprint("FUNCTION: intrp")

    tol = 1.0e-11
    def bisect(b,c):
        """Midpoint of [b, c] and the absolute tolerance there."""
        h = 0.5*(b+c)
        t = h*tol
        return (h,t)

    fc = common.de[0]
    fb = common.de[1]
    dprint("fc, fb= ",fc, fb)
    if(fc*fb>=0):
        return
    common.nord = common.ke[0]
    c = common.ce[0]
    b = common.ce[1]
    dprint("c,b= ",c,b)
    psq = 1.0/(b*b)
    strtdp(psq, om)
    s = c
    fs = fc
    h,t=bisect(b,c)

    while(abs(h-b)>=t): # Check for convergence
        dprint("fc,fb,c,b,h,t= ",fc,fb,c,b,h,t)

        if(abs(fb)<=abs(fc)):
            dprint("abs(fb) <= abs(fc): fg=fc, s=b, fs=fb")
            y = s
            fy = fs
            gg = c
            fg = fc
            s = b
            fs = fb
        else:
            dprint("abs(fb) > abs(fc): fg=fb, s=c, fs=fc")
            y = b
            fy = fb
            gg = b
            fg = fb
            s = c
            fs = fc

        if(fy != fs):
            b = (s*fy - y*fs)/(fy-fs) # Calculate b according to section formula
            if(abs(b-s)<t):
                b = s + math.copysign(t,gg-s)

            if((b-h)*(s-b)<0):
                b=h
        else:
            b = h # Case denominator becomes 0
        fb = detray(b,om,False)
        if(fg*fb < 0):
            dprint("fg*fb < 0")
            c = gg
            fc = fg
        else:
            dprint("fg*fb >= 0")
            c = s
            fc = fs
        h,t=bisect(b,c)

    dprint("Outside Loop: fc,fb,c,b,h,t= ",fc,fb,c,b,h,t)
    if(abs(fc)<abs(fb)):
        b=c
    dprint("After bisection")
    dprint("b= ",b)
    dprint("c= ",c)
    fb = detray(b,om,True)
    cp = b*(1.0-b/common.u)/om
    clin = b - cp*dom
    common.ctry = 5*common.cm -4*b - 2*(dom)*(common.um + 2*cp)
    common.ceps = max(abs(clin-common.ctry), common.ctry*common.tol)
    common.um = cp
    common.cm=b

    if(common.ctry==0):
        common.ctry = b - cp*dom
        common.ceps = max(abs(cp*dom) , common.ctry*common.tol)
