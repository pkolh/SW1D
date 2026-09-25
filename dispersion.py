"""The dispersion determinant, and the eigenfunction pass built on it."""
import math
import numpy as np
import struct

from common import common, dprint
from model import isInteger
from propagator import arg
from eigenfunctions import fixlov, fixray
from integrals import deriv

def detlov(cc ,w, ifeif):
    """"""
    dprint("===============")
    dprint("FUNCTION: detlov")
    dprint("ifeif= ",ifeif)
    p = 1.0/cc
    psq = p*p
    common.x[0][common.ist-1] = 1.0
    common.x[1][common.ist-1] = -common.fu[common.ist-1]*math.sqrt(psq - 1/common.vs[common.ist-1])
    i = common.ist
    common.scale[i-1]=0

    while(True):
        i = i - 1
        f1 = 1.0/common.fu[i-1]
        hb = psq - 1.0/common.vs[i-1]
        wd = w*common.d[i-1]
        cb,sb,fac = arg(wd,hb)
        common.scale[i-1] = common.scale[i] + fac
        fsb = f1*sb
        hsb = hb*sb*common.fu[i-1]
        common.x[0][i-1] = cb*common.x[0][i] + fsb*common.x[1][i]
        common.x[1][i-1] = hsb*common.x[0][i] + cb*common.x[1][i]
        if(i>common.noc):
            continue
        else:
            break

    de = common.x[1][common.noc-1]/(abs(common.x[1][common.noc-1]) + abs(common.x[0][common.noc-1]))
    if(ifeif):
        # undo arg()'s exponent scaling and normalise to the surface value
        xnorm = 1.0/common.x[0][common.noc-1]
        knt = common.noc -1
        for i in range(common.noc,common.ist+1):
            knt = knt + 1
            xfac=1
            if(common.scale[i-1]-common.scale[common.noc-1]!=0):
                xfac = math.exp(common.scale[i-1] - common.scale[common.noc-1])
            ls = i # Last i incase the model size is reduced
            common.x[0][i-1] = common.x[0][i-1]*xnorm*xfac
            common.x[1][i-1] = common.x[1][i-1]*xnorm*xfac
            if( abs(de) < 1.0e-4 and i > common.noc): #If stress dispacement vector is small and its not the first iteration
                pbsq = 1.0/common.vs[i-1]
                if(xfac < 1.0e-15 and psq >=pbsq): # If solution is no longer oscillatory
                    break

        if(abs(de)>=1.0e-4):
            x_updated = fixlov(w,p,common.x)

        hb = w*math.sqrt(psq - 1.0/common.vs[ls-1])
        fi1 = 0.5*common.x[0][ls-1]*common.x[0][ls-1]/hb
        fi2 = 0.5*common.x[1][ls-1]*common.x[1][ls-1]/(hb*common.fu[ls-1])
        # si1..si3 energy integrals; der[] log derivatives of c wrt rho, vs
        si1 = common.ro[ls-1]*fi1
        si2 = psq*common.fu[ls-1]*fi1 + fi2
        si3 = common.fu[ls-1]*fi1
        common.der[0][ls-1] = 0.5*(si2 - si1)
        common.der[1][ls-1] = si2
        for i in range(knt - 1, common.noc-1,-1):
            c0 = psq*common.fu[i-1] - common.ro[i-1]
            c1 = 0.5*(common.x[0][i-1]*common.x[1][i-1] - common.x[0][i]*common.x[1][i])/w
            c2 = 0.5*common.d[i-1]*( c0*common.x[0][i]*common.x[0][i] - common.x[1][i]*common.x[1][i]/common.fu[i-1] )
            fi1 = ( c2 - c1 )/c0
            fi2 = -c2 - c1
            sj1 = common.ro[i-1]*fi1
            sj2 = psq*common.fu[i-1]*fi1 + fi2

            si1=si1+sj1
            si2=si2+sj2

            si3=si3+common.fu[i-1]*fi1
            common.der[0][i-1] = 0.5*(sj2 - sj1)
            common.der[1][i-1] = sj2


        common.u = p*si3/si1
        per = 2*np.pi/w

        flan = si2/si1 - 1.0
        if(abs(flan)>=1.e-5):
            print("Problem with eigenfunctions: flan= ",flan," -- skipping this period, no excitation record written")
            common.wrote_record = False
            return de
        common.wrote_record = True

        if(common.iasc):
            intper = 0

            if(isInteger(per)):
                intper = 1


            strout = "{:3}" + 3*"{:13.7f} " + "{:15.7f}    {:4}    {:2d}"
            common.s10.write(strout.format(int(common.nord),per,cc,common.u,(cc*si3/w*w),int(ls),intper) + '\n')
            strout = 3*"{:15.7E}" + '\n'
            total_depth = 0
            for ic in range(1,ls+1):
                common.s10.write(strout.format(total_depth, common.x[0][ic-1], common.x[1][ic-1]))
                total_depth+=common.d[ic-1]

        fnorm = 1.0/(psq*si3)
        q = 0.0

        for i in range(common.noc,ls+1):
            common.der[0][i-1] = common.der[0][i-1]*fnorm
            common.der[1][i-1] = common.der[1][i-1]*fnorm
            q = q + common.der[1][i-1]*common.qb[i-1]
        gam = 0.50*w*q/cc
        if (common.irdep>ls):
            common.x[0][common.irdep-1]=0.0
            common.x[1][common.irdep-1]=0.0
        gb1 = w*common.x[1][common.irdep-1]/common.fu[common.irdep-1]
        strout="{} "*5
        binary_data = struct.pack('5d',w,cc,gam,common.x[0][common.irdep-1],gb1)
        common.iouf1.write(binary_data)
        if common.COLLECT is not None:
            common.COLLECT.add_love(nord=int(common.nord), w=w, c=cc, u=common.u,
                             gam=gam, i1=si1,
                             l1_rec=common.x[0][common.irdep-1],
                             l1_src=[common.x[0][int(common.idep[i])-1]
                                     for i in range(common.nsrce)],
                             gb1=gb1,
                             eig=(np.array([common.d[:ls],
                                            common.x[0][:ls], common.x[1][:ls]])
                                  if common.COLLECT.want_eigen else None),
                             der=(np.array([common.der[0][:ls],
                                            common.der[1][:ls]])
                                  if common.COLLECT.want_eigen else None))
        fact = 1.0/(math.sqrt(p*w)*si3*15.853309e-6)
        for i in range(1,common.nsrce+1):
            idd=int(common.idep[i-1])
            py1 = common.x[0][idd-1]*p*fact
            py2 = common.x[1][idd-1]*fact/common.fu[idd-1]
            if(idd>ls):
                py1=0
                py2=0
            strout="{} "*2
            binary_data = struct.pack('dd',py1,py2)
            common.iouf1.write(binary_data)
            if common.COLLECT is not None:
                common.COLLECT.add_love_py(py1, py2)
        per=2*np.pi/w
        if(q!=0):
            q=1/q
        if(common.iasc):
            strout="{}"+"{:12.6f} "+"{:12.6f} "*2 +"{:15.4f} "+"{:15.7E} "+"{:5} "*2
            common.iouf2.write(strout.format(int(common.nord),per,cc,common.u,q,flan,0,0))
            common.iouf2.write("\n")

    return de


def detray(cc , w , ifeif):
    """Rayleigh dispersion determinant at (cc, w); dispatches Love to detlov."""
    if(common.jcom==2):
        return(detlov(cc,w,ifeif))
    p = 1/cc
    ysav = 0
    psq = p*p
    r2 = 2*common.fu[common.ist-1]*p
    common.y[2] = math.sqrt(psq - 1/common.vp[common.ist-1])
    common.y[3] = - math.sqrt(psq - 1.0/common.vs[common.ist-1])
    common.y[0] = - (common.y[2]*common.y[3] + psq)/common.ro[common.ist-1]
    common.y[1] = r2*common.y[0] + p
    common.y[4] = common.ro[common.ist-1] - r2*(p + common.y[1])
    i = common.ist
    common.scale[i-1] = 0
    for j in range(1,6):
        common.ym[j-1][i-1]=common.y[j-1]
    #propagate up the layers
    while(True):
        i=i-1
        wd=w*common.d[i-1]
        ha=psq-1.0/common.vp[i-1]
        ca,sa,faca=arg(wd,ha)
        hb=psq-1.0/common.vs[i-1]
        cb,sb,facb=arg(wd,hb)
        common.scale[i-1]=common.scale[i]+faca+facb
        hbs = hb*sb
        has = ha*sa
        r1=1.0/common.ro[i-1]
        r2=2.0*common.fu[i-1]*p
        b1=r2*common.y[0]-common.y[1]
        g3=(common.y[4]+r2*(common.y[1]-b1))*r1
        g1=b1+(p*g3)
        g2=common.ro[i-1]*common.y[0]-p*(g1+b1)
        g1=g1*math.exp(-faca-facb)
        e1=cb*g2-hbs*common.y[2]
        e2=-sb*g2+cb*common.y[2]
        e3=cb*common.y[3]+hbs*g3
        e4=sb*common.y[3]+cb*g3
        common.y[2] = ca*e2 - has*e4
        common.y[3] = sa*e1 + ca*e3
        g3=ca*e4-sa*e2
        b1 = g1 - p*g3
        common.y[0] = (ca*e1 + has*e3 + p*(g1 + b1))*r1
        common.y[1] = r2*common.y[0] - b1
        common.y[4] = common.ro[i-1]*g3 - r2*(common.y[1] - b1)
        for j in range(1,6):
            common.ym[j-1][i-1]=common.y[j-1]
        if(i>common.noc):
            continue
        else:
            break
    if(common.noc!=1):
        ha=psq-1.0/common.vp[i-1]
        wd=w*d[0]
        ca,sa,faca=arg(wd,ha)
        common.y[0] = ca*common.y[2] - ha*sa*common.y[4]/common.ro[0]
        common.y[4] = ca*common.y[4] - common.ro[0]*sa*common.y[2]
        common.y[1] = common.y[4]
    de = common.y[4]/math.sqrt(common.y[0]*common.y[0] + common.y[1]*common.y[1])
    if(ifeif==0):
        return(de)
    else:
        ynorm = 1.0/common.ym[2][common.noc-1]
        if(common.noc!=1):
            common.y[0] = ca
            common.y[1] = common.ro[0]*sa
            ysav = common.y[1]/common.y[0]
            de1 = de
            de = min(abs(de1),abs(ynorm*common.ym[4][common.noc-1]/ysav - 1.0))
        common.y[0] = 0.0
        common.y[1] = -ynorm
        common.y[2] = 0.0
        common.y[3] = 0.0
        xfac = 1.0
        summ = 0.0
        i = common.noc

        while(1):
            xx1 = -common.ym[1][i-1]*common.y[0] - common.ym[2][i-1]*common.y[1] + common.ym[0][i-1]*common.y[3]
            xx2 = -common.ym[3][i-1]*common.y[0] + common.ym[1][i-1]*common.y[1] - common.ym[0][i-1]*common.y[2]
            xx3 = -common.ym[4][i-1]*common.y[1] - common.ym[1][i-1]*common.y[2] - common.ym[3][i-1]*common.y[3]
            xx4 =  common.ym[4][i-1]*common.y[0] - common.ym[2][i-1]*common.y[2] + common.ym[1][i-1]*common.y[3]


            common.x[0][i-1] = xx1*xfac
            common.x[1][i-1] = xx2*xfac
            common.x[2][i-1] = xx3*xfac
            common.x[3][i-1] = xx4*xfac

            ls = i
            if(i==common.ist):
                x_updated=np.copy(common.x)
                if(abs(de)>0.0001):
                    x_updated=fixray(ysav,w,p,common.x)

                dprint("deriv without fixray, Frequency = ",w)
                deriv(cc,w,ls,x_updated)
                return(de)
            else:
                if(i>=(common.noc+1) and abs(de)<=0.0001):
                    pbsq = 1.0/common.vs[i-1]
                    if(xfac<1e-15 and psq>=pbsq):
                        x_updated=np.copy(common.x)
                        if(abs(de)>0.0001):
                            x_updated=fixray(ysav,w,p,common.x)
                        dprint("deriv without fixray")
                        deriv(cc,w,ls,x_updated)
                        return (de)
            wd=w*common.d[i-1]
            ha=psq-1.0/common.vp[i-1]
            ca,sa,faca=arg(wd,ha)
            hb=psq-1.0/common.vs[i-1]
            cb,sb,facb=arg(wd,hb)

            dfac = math.exp(facb - faca)
            cb = dfac*cb
            sb = dfac*sb
            hbs = hb*sb
            has = ha*sa
            r2 = 2.0*p*common.fu[i-1]
            e2 = r2*common.y[1] - common.y[2]
            e3 = common.ro[i-1]*common.y[1] - p*e2
            e4 = r2*common.y[0] - common.y[3]
            e1 = common.ro[i-1]*common.y[0] - p*e4
            e6 = ca*e2 - sa*e1
            e8 = cb*e4 - sb*e3
            common.y[0] = (ca*e1 - has*e2 + p*e8)/common.ro[i-1]
            common.y[1] = (cb*e3 - hbs*e4 + p*e6)/common.ro[i-1]
            common.y[2] = r2*common.y[1] - e6
            common.y[3] = r2*common.y[0] - e8
            i = i + 1
            summ = summ + faca
            xfac = math.exp(common.scale[i-1] - common.scale[common.noc-1] + summ)
