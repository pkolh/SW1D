"""Summing modes into three-component displacement spectra."""
import numpy as np

from source import packed

OUT_SIGN = np.array([-1.0, +1.0, -1.0])

COMPONENTS = ("R", "T", "Z")

def _ray_basis(sol, phi):
    """Moment tensor projected on the ray basis at azimuth phi (radians)."""
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = c1*c1 - s1*s1, 2.0*c1*s1
    mzz, mxx, myy, mxz, myz, mxy = sol
    return dict(m_rr=0.5*(1+c2)*mxx + 0.5*(1-c2)*myy + s2*mxy,
                m_rd=c1*mxz + s1*myz,
                m_dd=mzz,
                m_td=-s1*mxz + c1*myz,
                m_rt=0.5*s2*(myy - mxx) + c2*mxy)

def seismograms(rayleigh, love, moment, distance, azimuth, cfg,
                isrc=0, t0=0.0, scale=1.0):
    """Sum modes into R, T, Z displacement spectra."""
    dist = np.atleast_1d(np.asarray(distance, dtype=float))
    phi = np.radians(float(azimuth))
    omega = 2*np.pi*cfg.frequencies
    nf, nr = len(omega), len(dist)

    p = _ray_basis(packed(moment)*1e-27, phi)
    acc = np.zeros((3, nr, nf), dtype=complex)

    for m in (rayleigh, love):
        if m is None:
            continue
        py = m.py[:, isrc, :]

        # bin each mode onto the output frequency grid
        idx = np.clip(np.searchsorted(omega, m.omega), 1, nf-1)
        ib = np.where(
            np.abs(m.omega - omega[idx-1]) <= np.abs(m.omega - omega[idx]),
            idx-1, idx)

        wvd = m.k[:, None]*dist[None, :]
        cwd, swd = np.cos(wvd), np.sin(wvd)
        atn = np.exp(-m.gamma[:, None]*dist[None, :])

        if m.wave == "rayleigh":
            g = (py[:, 0]*p["m_rr"] + py[:, 2]*p["m_dd"]) \
                - 1j*(py[:, 1]*p["m_rd"])
            rad = g[:, None]*(atn*m.r1_rec[:, None])*(swd + 1j*cwd)
            ver = g[:, None]*(atn*m.r2_rec[:, None])*(cwd - 1j*swd)
            for j in range(nr):
                np.add.at(acc[0, j], ib, rad[:, j])
                np.add.at(acc[2, j], ib, ver[:, j])
        else:
            g = (-py[:, 1]*p["m_td"]) - 1j*(py[:, 0]*p["m_rt"])
            tra = g[:, None]*(atn*m.l1_rec[:, None])*(cwd - 1j*swd)
            for j in range(nr):
                np.add.at(acc[1, j], ib, tra[:, j])

    fact = np.where(dist > 0, -np.sqrt(1e3/np.where(dist > 0, dist, 1.0)), 0.0)
    phase = np.exp(1j*(np.pi*0.25 + omega*t0))
    total = acc*phase[None, None, :]*fact[None, :, None]*scale

    return dict(frequency=cfg.frequencies, distance=dist,
                azimuth=float(azimuth),
                spectra=OUT_SIGN[:, None, None]*(-cfg.df)*np.conj(total),
                f0=cfg.f0, df=cfg.df, evdate="0000000000",
                components=COMPONENTS)

def trace(s, comp, irec, oversample=1):
    """One component at one receiver as a time series; returns (t, u)."""
    ci = COMPONENTS.index(comp) if isinstance(comp, str) else comp
    df, f0 = s["df"], s["f0"]
    ndc = int(round(f0/df))
    nfreq = len(s["frequency"])
    n = 2*(ndc + nfreq - 1)*int(oversample)
    full = np.zeros(n//2 + 1, dtype=complex)
    full[ndc:ndc + nfreq] = np.conj(s["spectra"][ci, irec, :])
    dt = (1.0/df)/n
    return np.arange(n)*dt, n*np.fft.irfft(full, n=n)
