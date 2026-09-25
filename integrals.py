"""Rayleigh energy integrals and phase-velocity partial derivatives."""
import math
import numpy as np
import struct

from common import common, dprint

def deriv(cc,w,ls,x):
    """Rayleigh energy integrals and phase-velocity partial derivatives."""
    dprint("===============")
    dprint("FUNCTION: deriv")
    p=1/cc
    psq=p*p
    ha=math.sqrt(psq-1/common.vp[ls-1])
    hb=math.sqrt(psq-1/common.vs[ls-1])
    c1 = (hb*x[0][ls-1] + p*x[1][ls-1])/(psq - ha*hb)
    c2 = (ha*x[1][ls-1] + p*x[0][ls-1])/(psq - ha*hb)
    c3 = common.ro[ls-1]*c1*c2*(hb/common.vp[ls-1] + ha/common.vs[ls-1])/(p*(ha + hb))
    t1 = common.ro[ls-1]*(ha*c1*c1 + hb*c2*c2 - 2.0*p*c1*c2)
    t2 = 0.5*common.ro[ls-1]*c1*c1/(common.vp[ls-1]*ha)
    t3 = 0.5*common.ro[ls-1]*c2*c2/(common.vs[ls-1]*hb) + t2
    si1 = t1 + t3
    si2 = 4.0*common.vs[ls-1]*psq*(t1 + c3) + t3
    si3 = 0.5*(si2 + si1)
    common.der[0][ls-1] = 0.5*(si2 - si1)
    common.der[1][ls-1] = t2
    common.der[2][ls-1] = si2 - t2
    i = ls
    while(True):
        i1=i
        i=i-1
        r2=2*common.fu[i-1]*p
        e4 = r2*x[0][i1-1] - x[3][i1-1]
        e1 = common.ro[i-1]*x[0][i1-1] - p*e4
        e2 = r2*x[1][i1-1] - x[2][i1-1]
        e3 = common.ro[i-1]*x[1][i1-1] - p*e2
        f4 = r2*x[0][i-1] - x[3][i-1]
        f1 = common.ro[i-1]*x[0][i-1] - p*f4
        f2 = r2*x[1][i-1] - x[2][i-1]
        f3 = common.ro[i-1]*x[1][i-1] - p*f2
        dh = 0.5*common.d[i-1]*w
        ha = psq - 1/common.vp[i-1]
        hb = psq - 1/common.vs[i-1]
        c1 = dh*(e1*e1 - ha*e2*e2)
        c2 = 0.50*(e1*e2 - f1*f2)
        c3 = dh*(e3*e3 - hb*e4*e4)
        c4 = 0.50*(e3*e4 - f3*f4)
        c5 = p*(e2*e4 - f2*f4)
        c6 = cc*(e1*e3 - f1*f3) - c5
        t1 = 2.0*(c5 + c2 + c4)/common.ro[i-1]
        t2 = (c2 - c1)/(common.ro[i-1]*common.vp[i-1]*ha)
        t3 = (c4 - c3)/(common.fu[i-1]*hb) + t2
        sj1 = t1 + t3
        sj2 = 4.0*common.vs[i-1]*psq*(t1 + c6/common.ro[i-1]) + t3
        si1 = si1 + sj1
        si2 = si2 + sj2
        si3 = si3 + 0.50*(sj2 + sj1) - (c1 + c3)/common.ro[i-1]
        common.der[0][i-1] = 0.50*(sj2 - sj1)
        common.der[1][i-1] = t2
        common.der[2][i-1] = sj2 - t2
        if(i>common.noc):
            continue
        else:
            break

    if(common.noc!=1):
        ha = (psq - 1.0/common.vp[0])/common.ro[0]
        c1 = 0.50*common.d[0]*w*(x[0][common.noc-1]*x[0][common.noc-1] - ha*x[2][common.noc-1]*x[2][common.noc-1]/common.ro[0])
        c2 = -0.50*x[0][common.noc-1]*x[2][common.noc-1]/common.ro[0]
        t2 = (c2 - c1)/(ha*common.vp[0])
        t1 = 2.0*common.ro[0]*c2 + t2
        si1 = si1 + t1
        si2 = si2 + t2
        si3 = si3 + 0.50*(t1 + t2) - common.ro[0]*c1
        common.der[0][0]=0.50*(t2 - t1)
        common.der[1][0]=t2
        common.der[2][0]=0.0
    common.u = cc*si3/si1
    flan = si1/si2 - 1.0
    if(abs(flan)>=1e-8):
        print("Problem with eigenfunction: flan = ",flan," -- skipping this period, no excitation record written")
        common.wrote_record = False
        return
    common.wrote_record = True
    if(common.irdep>ls):
        x[0][common.irdep-1]=0.0
        x[1][common.irdep-1]=0.0
        x[2][common.irdep-1]=0.0
        x[3][common.irdep-1]=0.0
    fnorm=1/si3
    for i in range(1,ls+1):
        for j in range(1,4):
            common.der[j-1][i-1]=common.der[j-1][i-1]*fnorm
    per=2*np.pi/w


    if(common.iasc==1):
        if(abs(per-int(per))<0.00001):
            intper=1
        else:
            intper=0
        common.dep[0]=0
        for ic in range(1,ls+1):
            common.dep[ic]=common.dep[ic-1]+common.d[ic-1]
        calc=w*w/(cc*si3)
        strout = "{:3}" + 3*"{:13.7f} " + "{:15.7f}    {:4}    {:2d}"
        common.s10.write(strout.format(int(common.nord),per,cc,common.u,calc,ls,intper))
        common.s10.write("\n")

        for ic in range(1,ls+1):
            strout="{:10.6E} "
            common.s10.write(strout.format(common.dep[ic-1]))
            for ir in range(1,5):
                common.s10.write(strout.format(x[ir-1][ic-1]))
            common.s10.write("\n")


    q=0
    for i in range(common.noc,ls+1):
        q=q+common.der[1][i-1]*common.qa[i-1]+common.der[2][i-1]*common.qb[i-1]
    gam = 0.50*w*q/cc
    size_prefix=32
    gb2 = w*(-x[0][common.irdep-1]/cc + x[3][common.irdep-1]/common.fu[common.irdep-1])
    gb1 = w*((1.0 - 2.0*common.vs[common.irdep-1]/common.vp[common.irdep-1])*x[1][common.irdep-1]/cc + x[2][common.irdep-1]/(common.ro[common.irdep-1]*common.vp[common.irdep-1]))
    binary_data=struct.pack('i',size_prefix)
    binary_data = struct.pack('7d', w,cc,gam,x[0][common.irdep-1],x[1][common.irdep-1],gb1,gb2)
    common.iouf1.write(binary_data)
    if common.COLLECT is not None:
        common.COLLECT.add_rayleigh(nord=int(common.nord), w=w, c=cc, u=common.u,
                             gam=gam, i1=si1,
                             r2_rec=x[0][common.irdep-1],
                             r1_rec=x[1][common.irdep-1],
                             r2_src=[x[0][int(common.idep[i])-1]
                                     for i in range(common.nsrce)],
                             r1_src=[x[1][int(common.idep[i])-1]
                                     for i in range(common.nsrce)],
                             gb1=gb1, gb2=gb2,
                             eig=(np.array([common.d[:ls], x[0][:ls], x[1][:ls],
                                            x[2][:ls], x[3][:ls]])
                                  if common.COLLECT.want_eigen else None),
                             der=(np.array([common.der[0][:ls],
                                            common.der[1][:ls],
                                            common.der[2][:ls]])
                                  if common.COLLECT.want_eigen else None))
    common.deriv_count+=1
    fact=p*math.sqrt(p*w)/(si3*15.853309e-6)
    for i in range(1,common.nsrce+1):
        idd=int(common.idep[i-1])
        py1=x[1][idd-1]*p*fact
        py2=x[3][idd-1]*fact/common.fu[idd-1]
        sig=common.ro[idd-1]*common.vp[idd-1]
        py3=-(x[2][idd-1]/sig + p*(1.0-2.0*common.fu[idd-1]/sig)*x[1][idd-1])*fact
        if(idd>ls):
            py1=0.0
            py2=0.0
            py3=0.0
        strout="{} "*3

        binary_data = struct.pack('3d',py1,py2,py3)
        common.iouf1.write(binary_data)
        if common.COLLECT is not None:
            common.COLLECT.add_rayleigh_py(py1, py2, py3)

    if(q!=0.0):
        q=1.0/q

    if(common.iasc==1):


        if(per<400 and round(per)%5==0 and abs(per-round(per))<0.00001):
            strout="{:.0f} "+"{:12.6f} "*3 +"{:12.4f} "+"{:16E} "+"{} "*2
            common.iouf2.write(strout.format(common.nord,per,cc,common.u,q,flan,ls,1))
            common.iouf2.write("\n")
            for j in range(1,4):
                for i in range(1,ls+1):
                    strout="{:.7E}"
                    common.iouf2.write(strout.format(common.der[j-1][i-1]))
                    common.iouf2.write(" ")
                    if(i%7==0 and i!=ls):
                        common.iouf2.write("\n")
                common.iouf2.write("\n")
        else:
            strout="{:.0f} "+"{:12.6f} "*3 +"{:12.4f} "+ "{:16E} "+"{} "*2
            common.iouf2.write(strout.format(common.nord,per,cc,common.u,q,flan,ls,0))
            common.iouf2.write("\n")
    return
