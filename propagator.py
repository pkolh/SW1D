"""Crossing a layer: propagator terms, starting depth, minor-vector recursion."""
from math import sin, cos, sqrt, exp

from common import common, dprint

def arg2(wd, h):
    """cos/sin (or cosh/sinh) pair for one layer, without the exponent scale."""
    c, s, _fac = arg(wd, h)
    return (c, s)

def arg(wd, h,cond=True):
    """Propagator terms across one layer of thickness wd/w."""
    hh = sqrt(abs(h))
    th = wd*hh
    if(th < 1.5e-14 and (cond or h>=0)):
        return (1.0, -wd, 0.0)
    if(h > 0):
        d = exp(-2.0*th)
        return (0.5*(1+d), -0.5*(1-d)/hh, th)
    return (cos(th), -1*sin(th)/hh, 0.0)

def strtdp(psq, w):
    """Starting layer for the upward recursion, into common.ist."""
    dprint("===============")
    dprint("FUNCTION: strtdp")
    # traverses VS2 to find the starting layer
    nm1=common.n-1
    for k in range(common.noc, common.n + 1): #the Fortran had common.noc+1 here
        j = common.n + common.noc - k #the Fortran had common.n+1 here
        if ((psq - (1.0 / common.vs[j - 1])) <= 0.0):
            break
    if (j < nm1):
        test = 0.0
        j=j+1
        knt=j-1
        for k_inner in range(j, nm1+1):
            knt+=1
            hb = sqrt(psq - (1.0 / common.vs[k_inner - 1]))
            test = test+common.d[k_inner - 1]*hb*w
            if (test > 40.0):
                break
        common.ist = knt+1
        dprint("common.ist= ",common.ist)
        return
    else:
        common.ist = common.n
        dprint("common.ist= ",common.ist)
        return

def propagate(i,p,kount,psq,w):
    """Propagate the minor vector up through layer i, counting crossings."""
    dprint("===============")
    dprint("FUNCTION: propagate")
    tpi=6.28318
    while(True):
        dprint("OUTER LOOP RESTARTED")
        reset=1
        i=i-1
        r1=1.0/common.ro[i-1]
        r2=2.0*common.fu[i-1]*p
        b1=r2*common.yknt[0]-common.yknt[1]
        g3=(common.yknt[4]+r2*(common.yknt[1]-b1))*r1
        g1=b1+(p*g3)
        g2=common.ro[i-1]*common.yknt[0]-p*(g1+b1)
        ha=psq-1.0/common.vp[i-1]
        hb=psq-1.0/common.vs[i-1]
        summ=0.0
        dels=0.0
        delm=common.d[i-1]

        if(hb<0.0):
            delm=tpi*0.1/(w*(float(sqrt(-hb))))
        if(ha<0.0):
            delm=tpi*0.1/(w*(float(sqrt(-hb))+float(sqrt(-ha))))


        while(True):
            dprint("INNER LOOP RESTARTED")
            if(reset==1):
                dell=delm
                isplit=0
            reset=1
            if((summ+dell)>common.d[i-1]):
                dell=common.d[i-1]-summ
            if(dels!=dell):
                dels=dell
                wd=w*dell
                ca,sa,faca=arg(wd,ha)
                cb,sb,facb=arg(wd,hb)
                hbs=hb*sb
                has=ha*sa

            g1=g1*exp(-faca-facb)
            e1=cb*g2-hbs*common.yknt[2]
            e2=-sb*g2+cb*common.yknt[2]
            e3=cb*common.yknt[3]+hbs*g3
            e4=sb*common.yknt[3]+cb*g3

            common.xknt[2]=ca*e2-has*e4
            common.xknt[3]=sa*e1+ca*e3
            g2=ca*e1+has*e3
            g3=ca*e4-sa*e2
            b1=g1-p*g3
            common.xknt[0]=(g2+p*(b1+g1))*r1
            common.xknt[1]=r2*common.xknt[0]-b1
            common.xknt[4]=common.ro[i-1]*g3-r2*(common.xknt[1]-b1)
            if(common.xknt[4]*common.yknt[4]<=0):
                t1=common.yknt[2]-common.yknt[3]
                t2=common.xknt[2]-common.xknt[3]
                if(t1*t2>0):
                    tes=t1*(common.yknt[4]-common.xknt[4])
                    dprint("tes= ",tes)
                    if(tes<0.0):
                        kount=kount+1
                        dprint("kount= ",kount)
                    if(tes>0.0):
                        kount=kount-1
                        dprint("kount= ",kount)
                    for j in range(1,6):
                        common.yknt[j-1]=common.xknt[j-1]
                    summ=summ+dell
                    if(summ<common.d[i-1]):
                        continue

                else:
                    dell=0.5*dell
                    isplit+=1
                    b1=r2*common.yknt[0]-common.yknt[1]
                    g3=(common.yknt[4]+r2*(common.yknt[1]-b1))*r1
                    g1=b1+p*g3
                    g2=common.ro[i-1]*common.yknt[0]-p*(g1+b1)
                    if(isplit<100):
                        reset=0
                        continue
                    print("propagate: mode count did not settle in this "
                          "layer after 100 bisections")
                    quit()
            else:
                x5p=common.yknt[4]-wd*(common.ro[i-1]*(common.yknt[3]-common.yknt[2])-4.0*common.fu[i-1]*psq*common.yknt[3]*(1.0-common.vs[i-1]/common.vp[i-1]))
                if((common.xknt[4]*x5p)<0):
                    dell=0.5*dell
                    isplit+=1
                    b1=r2*common.yknt[0]-common.yknt[1]
                    g3=(common.yknt[4]+r2*(common.yknt[1]-b1))*r1
                    g1=b1+p*g3
                    g2=common.ro[i-1]*common.yknt[0]-p*(g1+b1)
                    if(isplit<100):
                        reset=0
                        continue
                    print("propagate: mode count did not settle in this "
                          "layer after 100 bisections")
                    quit()
                else:
                    for j in range(1,6):
                        common.yknt[j-1]=common.xknt[j-1]
                    summ=summ+dell
                    if(summ<common.d[i-1]):
                        continue

            dprint("Innermost BREAK executed")
            break
        if(i>common.noc):
            continue
        else:
            break

    if(common.noc!=1):
        #propagate through ocean layer
        common.yknt[0]=common.yknt[2]
        common.yknt[1]=common.yknt[4]
        r1=1.0/common.ro[0]
        ha=psq-1.0/common.vp[0]
        summ=0.0
        dels=0.0
        dell=common.d[0]
        if(ha<0.0):
            dell=tpi*0.1/(w*float(sqrt(-ha)))
        while(True):
            if((summ+dell)>common.d[0]):
                dell=common.d[0]-summ
            if(dels!=dell):
                dels=dell
                wd=w*dell
                ca,sa,faca=arg(wd,ha)
                rsa=common.ro[0]*sa
                hsa=ha*sa*r1
            x1=ca*common.yknt[0]-hsa*common.yknt[1]
            x2=ca*common.yknt[1]-rsa*common.yknt[0]
            if(x2*common.yknt[1]>0.0):
                common.yknt[0]=x1
                common.yknt[1]=x2
                summ=summ+dell
                if(summ<common.d[0]):
                    continue
                else:
                    common.yknt[4]=common.yknt[1]
                    root=float(sqrt(common.yknt[0]*common.yknt[0]+common.yknt[1]*common.yknt[1]))
                    de=common.yknt[4]/root
                    dprint("End of propagate")
                    dprint("kount= ",kount)
                    dprint("de= ",de)
                    dprint("This value is passed all the way to whatever calls detk- main function or foo")
                    return(kount,de)
            else:
                if(x1*common.yknt[0]>0):
                    tes=x1*(common.yknt[1]-x2)
                    if(tes<0.0):
                        kount=kount+1
                    if(tes>0.0):
                        kount=kount-1
                    common.yknt[0]=x1
                    common.yknt[1]=x2
                    summ=summ+dell
                    if(summ<common.d[0]):
                        continue
                    else:
                        common.yknt[4]=common.yknt[1]
                        root=float(sqrt(common.yknt[0]*common.yknt[0]+common.yknt[1]*common.yknt[1]))
                        de=common.yknt[4]/root
                        dprint("End of propagate")
                        dprint("kount= ",kount)
                        dprint("de= ",de)
                        dprint("This value is passed all the way to whatever calls detk- main function or foo")
                        return(kount,de)

                else:
                    dell=0.5*dell
                    continue
        common.yknt[4]=common.yknt[1]
    dprint("Propagate: y[1]= ",common.yknt[1])
    root=float(sqrt(common.yknt[0]*common.yknt[0]+common.yknt[1]*common.yknt[1]))
    de=common.yknt[4]/root
    dprint("End of propagate")
    dprint("kount= ",kount)
    dprint("de= ",de)
    dprint("This value is passed all the way to whatever calls detk- main function or foo")
    return(kount,de)

def solve_coef(ii,x,b):
    """Coefficient matching the two half-space solutions at the starting layer."""
    dprint("===============")
    dprint("FUNCTION: solve_coef")
    xx=0.0
    b22=0.0
    dprint("ii= ",ii)
    for k in range(1,5):
        xx=xx+x[k-1][ii-1]*x[k-1][ii-1]
        dprint("xx done")
        b22=b22+b[1][k-1]*b[1][k-1]
    dprint("xx1= ",xx)
    f=sqrt(b22/xx)
    dprint("f= ",f)
    xb1=0.0
    xb2=0.0
    b12=0.0
    for k in range(1,5):
        xb1=xb1+x[k-1][ii-1]*b[0][k-1]*f
        xb2=xb2+x[k-1][ii-1]*b[1][k-1]*f
        b12=b12+b[0][k-1]*b[1][k-1]
    det=b22*b22-xb2*xb2
    dprint("det= ",det)
    coef=(xb2*xb1-b22*b12)/det
    dprint(coef)
    return(coef)
