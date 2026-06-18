"""Make the integration's pure modules importable without loading Home Assistant.

``models.py`` and ``data_processing.py`` depend only on numpy/pandas, but importing
them through the normal package path (``custom_components.ha_power_predictor.models``)
would execute the package ``__init__.py``, which imports Home Assistant. Adding the
package directory itself to ``sys.path`` lets these tests import them as standalone
top-level modules, so the pure test job needs no Home Assistant install.
"""

import pathlib
import sys

_PKG = pathlib.Path(__file__).resolve().parents[2] / "custom_components" / "ha_power_predictor"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
