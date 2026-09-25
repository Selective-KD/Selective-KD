from .abkd import ABKD
from .betakd import BetaKD
from .dkd import DKD
from .fusions import ABKDUGKD, DKDUGKD, LSDUGKD, WLSUGKD, WTTMUGKD
from .hinton import Hinton
from .lsd import LSD
from .lud import LUD
from .mcd import MCD
from .ours import UGKD
from .rwkd import RWKD
from .tgeo import TGeo
from .ubkd import UBKD
from .wls import WLS
from .wttm import WTTM

REGISTRY = {cls.name: cls for cls in (UGKD, Hinton, LUD, MCD, UBKD, DKD, BetaKD, TGeo, WLS, RWKD, ABKD, LSD, WTTM,
                                       DKDUGKD, ABKDUGKD, LSDUGKD, WTTMUGKD, WLSUGKD)}
NAMES = sorted(REGISTRY)


def build(name, T, overrides=None):
    cls = REGISTRY[name]
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(cls.DEFAULTS))
    if unknown:
        raise ValueError("%s has no parameter %s; its parameters are %s" % (name, unknown, sorted(cls.DEFAULTS)))
    return cls(T, {**cls.DEFAULTS, **overrides})


def defaults_text():
    return "\n".join("  %-10s %s" % (n, REGISTRY[n].DEFAULTS or "-") for n in NAMES)
