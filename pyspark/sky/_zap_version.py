# -*- coding: utf-8 -*-

__version2__ = '2.2.dev'
__version1__ = '1.1.dev'
#__date__ = '2017/09/08'
__description__ = ('ZAP (the Zurich Atmosphere Purge) is a high precision sky'
                   ' subtraction tool.')

try:
    from ._githash import __githash__, __dev_value__
    if '.dev' in __version__:
        __version__ += __dev_value__
except Exception:
    pass
