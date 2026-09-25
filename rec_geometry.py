"""Receiver distances and the source-to-receiver azimuth."""
import math
import numpy as np

from inputs import _need

# geocentric: the azimuth and distance a pair of lat/lon corners imply

def bjdaz2(zlat0, alon0, zlat, alon, lattype):
    """Distance and both azimuths between two points, all in radians."""
    pii = math.pi
    twopi = 2*pii

    if lattype == 1:                 # geocentric latitude
        alat0 = pii*0.5 - zlat0
        alat = pii*0.5 - zlat
    else:                            # geocentric colatitude
        alat0 = zlat0
        alat = zlat

    st0 = np.sin(alat0)
    ct0 = np.cos(alat0)
    ct1 = np.cos(alat)
    s0c1 = st0*ct1
    st1 = np.sin(alat)
    s1c0 = st1*ct0

    dlon = alon - alon0
    sdlon = np.sin(dlon)
    cdlon = np.cos(dlon)

    # epicentral distance
    cdelt = st0*st1*cdlon + ct0*ct1
    b = s0c1 - s1c0*cdlon
    a = st1*sdlon
    sdelt = np.sqrt(b**2 + a**2)
    ddelt = np.arctan2(sdelt, cdelt)

    # azimuth
    aze = 0.0
    if sdelt != 0.0:
        aze = np.arctan2(a, b)

    # back azimuth
    a = -sdlon*st0
    b = s1c0 - s0c1*cdlon
    azs = pii
    if sdelt != 0.0:
        azs = np.arctan2(a, b)

    # bring both into 0 < azimuth < 2 pi
    if aze < 0.0:
        aze += twopi
    if azs < 0.0:
        azs += twopi
    return ddelt, aze, azs

def glat(hlat):
    """Geographic latitude to geocentric, both in radians, north positive."""
    halfpi = 1.570796
    polefac = 0.010632
    elfac = 0.993277

    if halfpi - abs(hlat) >= 0.05:
        return math.atan(elfac*math.sin(hlat)/math.cos(hlat))
    return hlat/elfac - math.copysign(polefac, hlat)     # near the pole


DEGRAD = 0.017453293

KMPERDEG = 111.19493

AZ_KEYS = ("azimuth", "az")

def specs_azimuth(s):
    """The azimuth written straight into the file, in degrees, or None."""
    vals = {k: float(s[k]) for k in AZ_KEYS if k in s}
    if not vals:
        return None
    if len(set(vals.values())) > 1:
        raise ValueError("azimuth and az disagree: "
                         + ", ".join(f"{k}={v:g}" for k, v in vals.items()))
    return next(iter(vals.values())) % 360.0

def _corner_geometry(s):
    """(d1, d2, azimuth) from the event to the two glat/glon corners."""
    _need(s, ("slatd", "slond", "glatmin", "glonmin", "glatmax", "glonmax"),
          "the receiver azimuth")
    hpi = math.pi/2.0
    slato = hpi - glat(s["slatd"]*DEGRAD)
    slono = s["slond"]*DEGRAD

    d1, az1, _ = bjdaz2(slato, slono,
                        hpi - glat(s["glatmin"]*DEGRAD),
                        s["glonmin"]*DEGRAD, 0)
    d2, az2, _ = bjdaz2(slato, slono,
                        hpi - glat(s["glatmax"]*DEGRAD),
                        s["glonmax"]*DEGRAD, 0)
    return d1, d2, 0.5*(az1 + az2)/DEGRAD

def specs_geometry(s):
    """Distances (km) and azimuth (deg) from an inputs dict."""
    given_az = specs_azimuth(s)

    if str(s.get("dist_calc_choice", "d")).startswith("l"):
        if given_az is not None:
            raise ValueError("azimuth is only accepted with dist_calc_choice d;"
                             " with 'l' the glat/glon corners set it")
        d1, d2, azav = _corner_geometry(s)
        nx = int(s.get("nx", 2))
        r0, r1 = d1*KMPERDEG/DEGRAD, d2*KMPERDEG/DEGRAD
        return np.linspace(r0, r1, max(nx, 1)), azav

    if given_az is not None:
        azav = given_az
    else:
        try:
            azav = _corner_geometry(s)[2]
        except ValueError as e:
            raise ValueError(f"{e}; or write the azimuth down directly, "
                             f"'azimuth <deg>'") from None
    _need(s, ("dist_reci", "dist_recf", "dx"), "the receiver distances")
    r0, r1, dx = s["dist_reci"], s["dist_recf"], s["dx"]
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if r1 < r0:
        raise ValueError(f"dist_recf ({r1}) is less than dist_reci ({r0})")
    # truncation drops the last receiver when (r1-r0)/dx lands just under an
    # integer: 250 to 250.7 by 0.1 gave 7 ending at 250.6, not 8 at 250.7
    steps = (r1 - r0)/dx
    dist = r0 + dx*np.arange(int(math.floor(steps + 1e-9*max(1.0, steps))) + 1)
    return dist, azav
