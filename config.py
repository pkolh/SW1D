"""The run configuration: model, band, depths, mode range."""
import numpy as np
from dataclasses import dataclass, fields

@dataclass
class RunConfig:
    """The model and the run: layers, band, depths, mode range."""
    thickness: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    qbeta: np.ndarray = None
    qalpha: np.ndarray = None
    model_name: str = "model"

    f0: float = 0.005
    df: float = 0.0005
    nfreq: int = 191

    source_depths: tuple = (10.0,)
    receiver_depth: float = 0.0

    # inclusive; mode_max < 0 means 'as many as exist'
    mode_min: int = 0
    mode_max: int = 5

    # phase-velocity search bounds, km/s; 0 lets the solver choose
    cmin: float = 0.0
    cmax: float = 0.0

    # flattening, and reference period for the Q dispersion correction
    flatten: bool = False
    tref: float = 1.0

    # periods (s) at which a solution is forced in addition to the grid
    ensure_periods: tuple = ()

    mode_files: bool = False
    wave: str = "rayleigh"          # set per solve

    def __post_init__(self):
        """Coerce the layer arrays to float, fill absent Q, and validate."""
        n = len(self.thickness)
        if self.qbeta is None:
            self.qbeta = np.zeros(n)
        if self.qalpha is None:
            self.qalpha = np.zeros(n)
        for nm in ("thickness", "vp", "vs", "rho", "qbeta", "qalpha"):
            setattr(self, nm, np.asarray(getattr(self, nm), dtype=float))
            if len(getattr(self, nm)) != n:
                raise ValueError(f"{nm} has {len(getattr(self, nm))} entries, "
                                 f"thickness has {n}")
        self.source_depths = tuple(float(d) for d in self.source_depths)
        self.ensure_periods = tuple(sorted(float(t) for t in self.ensure_periods))
        self.validate()

    @property
    def nlayers(self):
        """Number of layers, including the half-space."""
        return len(self.thickness)

    @property
    def frequencies(self):
        """The output frequency grid, Hz."""
        return self.f0 + np.arange(self.nfreq)*self.df

    @property
    def fmax(self):
        """Highest frequency on the grid, Hz."""
        return self.f0 + (self.nfreq - 1)*self.df

    def depth_top(self):
        """Depth to the top of each layer, km."""
        return np.concatenate([[0.0], np.cumsum(self.thickness)[:-1]])

    def validate(self):
        """Check the settings are usable. Call again after changing a field."""
        if int(self.nfreq) < 1:
            raise ValueError(f"nfreq must be at least 1, got {self.nfreq}")
        if self.df <= 0:
            raise ValueError(f"df must be positive, got {self.df}")
        if self.f0 < 0:
            raise ValueError(f"f0 must not be negative, got {self.f0}")
        if len(self.source_depths) > 4:
            raise ValueError(f"at most 4 source depths (kernel limit lsd=4), "
                             f"got {len(self.source_depths)}")
        if not self.source_depths:
            raise ValueError("at least one source depth is needed")
        if any(d < 0 for d in self.source_depths):
            raise ValueError(f"source depths must not be negative, "
                             f"got {self.source_depths}")
        if self.receiver_depth < 0:
            raise ValueError(f"receiver depth must not be negative, "
                             f"got {self.receiver_depth}")
        if self.mode_min < 0:
            raise ValueError(f"the lowest mode must be at least 0 "
                             f"(0 is the fundamental), got {self.mode_min}")
        if 0 <= self.mode_max < self.mode_min:
            raise ValueError(f"mode range {self.mode_min}-{self.mode_max} is "
                             f"empty; the highest mode must be at least the "
                             f"lowest, or negative for 'every branch'")
        if self.cmin < 0 or self.cmax < 0:
            raise ValueError("cmin and cmax must not be negative "
                             "(0 lets the solver choose)")
        if self.cmin and self.cmax and self.cmax <= self.cmin:
            raise ValueError(f"cmax ({self.cmax}) must exceed cmin ({self.cmin})")
        if self.nlayers > 3000:
            raise ValueError(f"at most 3000 layers (kernel limit lyrs=3000), "
                             f"got {self.nlayers}")
        return self

    def describe(self):
        """One-paragraph summary of the run, for the log and the terminal."""
        return (f"{self.model_name}: {self.nlayers} layers, "
                f"f {self.f0:g}-{self.fmax:g} Hz "
                f"(df={self.df:g}, {self.nfreq} freqs, record {1/self.df:g} s), "
                f"modes {self.mode_min}-{self.mode_max}, "
                f"source {self.source_depths} km, "
                f"receiver {self.receiver_depth} km, "
                f"flattening {'on' if self.flatten else 'off'}")
