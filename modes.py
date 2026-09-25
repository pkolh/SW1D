"""The solved modes, and the object that collects them."""
import numpy as np
from dataclasses import dataclass, fields

@dataclass
class Modes:
    """One wave type: flat arrays, one entry per (branch, frequency)."""
    wave: str
    branch: np.ndarray
    omega: np.ndarray
    c: np.ndarray
    u: np.ndarray
    gamma: np.ndarray
    i1: np.ndarray
    py: np.ndarray
    eig: list = None
    der: list = None
    l1_rec: np.ndarray = None
    l1_src: np.ndarray = None
    r1_rec: np.ndarray = None
    r2_rec: np.ndarray = None
    r1_src: np.ndarray = None
    r2_src: np.ndarray = None
    gb1: np.ndarray = None
    gb2: np.ndarray = None
    source_depths: tuple = ()
    receiver_depth: float = 0.0
    nbranch: int = 0

    @property
    def nmodes(self):
        """Number of (branch, frequency) solutions held."""
        return len(self.omega)

    @property
    def frequency(self):
        """Frequency of each solution, Hz."""
        return self.omega/(2*np.pi)

    @property
    def period(self):
        """Period of each solution, s."""
        return 2*np.pi/self.omega

    @property
    def k(self):
        """Wavenumber, 1/km."""
        return self.omega/self.c

    def branches(self):
        """Sorted list of the branch numbers present."""
        return sorted(set(int(b) for b in self.branch))

    def select(self, branches):
        """A new Modes holding only the given branch number(s)."""
        want = np.atleast_1d(np.asarray(branches))
        m = np.isin(self.branch, want)
        if not m.any():
            raise ValueError(f"no modes on branch(es) {branches}")
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray) and v.shape[:1] == self.branch.shape:
                out[f.name] = v[m]
            elif f.name in ("eig", "der") and isinstance(v, list) \
                    and len(v) == len(m):
                out[f.name] = [e for e, keep in zip(v, m) if keep]
            else:
                out[f.name] = v
        return Modes(**out)

    def sorted_branch(self, n):
        """Indices of branch n, in order of increasing frequency."""
        idx = np.flatnonzero(self.branch == n)
        return idx[np.argsort(self.omega[idx])]

    def __repr__(self):
        """Wave type, solution count and branch range."""
        b = self.branches()
        return (f"Modes({self.wave}, {self.nmodes} modes, "
                f"branches {b[0] if b else '-'}..{b[-1] if b else '-'}, "
                f"T {self.period.min():.1f}-{self.period.max():.1f} s)")

class Collector:
    """Accumulates solutions as the solver produces them."""

    def __init__(self, want_eigen=False):
        """One row per solution; want_eigen retains the eigenfunction profiles."""
        self.want_eigen = bool(want_eigen)
        self._rows = []
        self._cur = None
        self._branch = 0
        self.wave = None
        self.nbranch = 0
        self.source_depths = ()
        self.receiver_depth = 0.0

    def begin(self, cfg, jcom, nbran):
        """Start a run: record the wave type, branch count and source depths."""
        self.wave = "rayleigh" if jcom == 1 else "love"
        self.nbranch = nbran
        self.source_depths = tuple(sorted(cfg.source_depths))
        self.receiver_depth = cfg.receiver_depth

    def begin_branch(self, nb):
        """Start branch nb; subsequent rows belong to it."""
        self._branch = nb

    def add_love(self, **kw):
        """Add one Love solution."""
        kw["branch"] = self._branch
        kw["py"] = []
        self._cur = kw
        self._rows.append(kw)

    def add_love_py(self, py1, py2):
        """Add the excitation coefficients for the current Love solution."""
        self._cur["py"].append((py1, py2))

    def add_rayleigh(self, **kw):
        """Add one Rayleigh solution."""
        kw["branch"] = self._branch
        kw["py"] = []
        self._cur = kw
        self._rows.append(kw)

    def add_rayleigh_py(self, py1, py2, py3):
        """Add the excitation coefficients for the current Rayleigh solution."""
        self._cur["py"].append((py1, py2, py3))

    def end(self):
        """End of run. No finalisation needed."""
        pass

    def to_modes(self):
        """Pack the collected rows into a Modes object."""
        r = self._rows
        if not r:
            raise RuntimeError("solver produced no modes -- check the band, "
                               "the mode range and the model")
        g = lambda k: np.array([row[k] for row in r], dtype=float)

        # si1 is not A&R's I1: Love is out by 2, Rayleigh by 2*w
        si1 = g("i1")
        w = g("w")
        i1 = si1/2.0 if self.wave == "love" else si1/(2.0*w)

        shared = dict(wave=self.wave, branch=g("branch"), omega=w,
                      c=g("c"), u=g("u"), gamma=g("gam"), i1=i1,
                      py=np.array([row["py"] for row in r], dtype=float),
                      eig=[row.get("eig") for row in r],
                      der=[row.get("der") for row in r],
                      source_depths=self.source_depths,
                      receiver_depth=self.receiver_depth,
                      nbranch=self.nbranch)
        if self.wave == "love":
            return Modes(l1_rec=g("l1_rec"),
                         l1_src=np.array([row["l1_src"] for row in r]),
                         gb1=g("gb1"), **shared)
        return Modes(r1_rec=g("r1_rec"), r2_rec=g("r2_rec"),
                     r1_src=np.array([row["r1_src"] for row in r]),
                     r2_src=np.array([row["r2_src"] for row in r]),
                     gb1=g("gb1"), gb2=g("gb2"), **shared)
