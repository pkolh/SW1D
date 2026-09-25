"""The source: fault angles or a moment tensor."""
import numpy as np

def double_couple(strike, dip, rake, m0=1.0):
    """Double couple from the A&R angles (deg), in (North, East, Down)."""
    p, d, l = np.radians([strike, dip, rake])

    nu = np.array([-np.sin(d)*np.sin(p),
                    np.sin(d)*np.cos(p),
                   -np.cos(d)])
    u = np.array([np.cos(l)*np.cos(p) + np.cos(d)*np.sin(l)*np.sin(p),
                  np.cos(l)*np.sin(p) - np.cos(d)*np.sin(l)*np.cos(p),
                 -np.sin(l)*np.sin(d)])
    return m0*(np.outer(nu, u) + np.outer(u, nu))

def explosion(m0=1.0):
    """Isotropic source."""
    return m0*np.eye(3)

def packed(m):
    """[Mzz Mxx Myy Mxz Myz Mxy] = [M_DD M_NN M_EE M_ND M_ED M_NE]."""
    m = np.asarray(m, dtype=float)
    return np.array([m[2, 2], m[0, 0], m[1, 1], m[0, 2], m[1, 2], m[0, 1]])
