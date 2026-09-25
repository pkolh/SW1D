"""The nine-component Green's tensor, Aki & Richards (7.146)/(7.147)."""
import numpy as np

def _bin_index(omega_modes, omega_grid):
    """Nearest frequency bin for each mode."""
    idx = np.clip(np.searchsorted(omega_grid, omega_modes), 1, len(omega_grid)-1)
    left = np.abs(omega_modes - omega_grid[idx-1])
    right = np.abs(omega_modes - omega_grid[idx])
    return np.where(left <= right, idx-1, idx)

def _propagator(k, gamma, r):
    """sqrt(2/(pi k r)) exp[i(k r + pi/4)] exp(-gamma r)."""
    return (np.sqrt(2.0/(np.pi*k[:, None]*r[None, :]))
            * np.exp(1j*(k[:, None]*r[None, :] + np.pi/4.0))
            * np.exp(-gamma[:, None]*r[None, :]))

def _propagator_hankel(k, gamma, r):
    """i H_0^(1)(k r) exp(-gamma r) -- the exact form _propagator approximates."""
    try:
        from scipy.special import hankel1
    except ImportError:
        raise ImportError("the Hankel form of the Green's tensor needs scipy; "
                          "green_tensor() itself does not")
    return (1j*hankel1(0, k[:, None]*r[None, :])
            * np.exp(-gamma[:, None]*r[None, :]))

def green_tensor(rayleigh=None, love=None, distance=None, azimuth=0.0,
                 isrc=0, scale=1.0, omega=None, hankel=False):
    """Assemble the 9-component Green's tensor, A&R (7.146) + (7.147)."""
    prop = _propagator_hankel if hankel else _propagator
    if rayleigh is None and love is None:
        raise ValueError("need at least one of rayleigh=, love=")
    if distance is None:
        raise ValueError("distance is required")

    r = np.atleast_1d(np.asarray(distance, dtype=float))
    phi = np.radians(np.atleast_1d(np.asarray(azimuth, dtype=float)))
    if phi.size == 1 and r.size > 1:
        phi = np.repeat(phi, r.size)
    if r.size != phi.size:
        raise ValueError("distance and azimuth must have the same length")
    if np.any(r <= 0):
        raise ValueError("the far-field expansion needs r > 0")

    # without an omega= the modes set the grid themselves
    ref = rayleigh if rayleigh is not None else love
    if omega is None:
        omega = np.unique(np.concatenate(
            [m.omega for m in (rayleigh, love) if m is not None]))
    omega = np.asarray(omega, dtype=float)

    nf, nr = len(omega), len(r)
    Gr = np.zeros((nf, nr, 3, 3), dtype=complex)
    Gl = np.zeros((nf, nr, 3, 3), dtype=complex)
    cph, sph = np.cos(phi), np.sin(phi)

    if rayleigh is not None:
        m = rayleigh
        amp = (1.0/(8.0*m.c*m.u*m.i1))[:, None]*prop(m.k, m.gamma, r)

        # first letter receiver, second source; h = r1 horizontal, v = r2 vertical
        hh = (m.r1_rec*m.r1_src[:, isrc])[:, None]
        hv = (m.r1_rec*m.r2_src[:, isrc])[:, None]
        vh = (m.r2_rec*m.r1_src[:, isrc])[:, None]
        vv = (m.r2_rec*m.r2_src[:, isrc])[:, None]

        # (7.146): the azimuth dependence, component by component
        b = np.empty((m.nmodes, nr, 3, 3), dtype=complex)
        b[:, :, 0, 0] = hh*(cph*cph)[None, :]
        b[:, :, 0, 1] = hh*(cph*sph)[None, :]
        b[:, :, 0, 2] = -1j*hv*cph[None, :]
        b[:, :, 1, 0] = hh*(sph*cph)[None, :]
        b[:, :, 1, 1] = hh*(sph*sph)[None, :]
        b[:, :, 1, 2] = -1j*hv*sph[None, :]
        b[:, :, 2, 0] = 1j*vh*cph[None, :]
        b[:, :, 2, 1] = 1j*vh*sph[None, :]
        b[:, :, 2, 2] = vv*np.ones(nr)[None, :]

        np.add.at(Gr, _bin_index(m.omega, omega), b*amp[:, :, None, None])

    if love is not None:
        m = love
        amp = ((m.l1_rec*m.l1_src[:, isrc])/(8.0*m.c*m.u*m.i1))[:, None] \
            * prop(m.k, m.gamma, r)

        # (7.147): Love fills the horizontal block and nothing else
        b = np.zeros((m.nmodes, nr, 3, 3), dtype=complex)
        b[:, :, 0, 0] = (sph*sph)[None, :]
        b[:, :, 0, 1] = (-sph*cph)[None, :]
        b[:, :, 1, 0] = (-sph*cph)[None, :]
        b[:, :, 1, 1] = (cph*cph)[None, :]

        np.add.at(Gl, _bin_index(m.omega, omega), b*amp[:, :, None, None])

    Gr *= scale
    Gl *= scale
    return dict(omega=omega, distance=r, azimuth=phi, G=Gr + Gl,
                G_rayleigh=Gr, G_love=Gl,
                source_depth=(ref.source_depths[isrc]
                              if ref.source_depths else 0.0),
                receiver_depth=ref.receiver_depth)

def green_tensor_hankel(rayleigh=None, love=None, distance=None, azimuth=0.0,
                        isrc=0, scale=1.0, omega=None):
    """green_tensor() with the exact H_0^(1)(kr), not its asymptotic form."""
    return green_tensor(rayleigh=rayleigh, love=love, distance=distance,
                        azimuth=azimuth, isrc=isrc, scale=scale, omega=omega,
                        hankel=True)
