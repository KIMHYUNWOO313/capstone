import timesfm
from autogluon.timeseries import TimeSeriesPredictor  # noqa: F401

import web_app.app  # noqa: F401


print("timesfm", getattr(timesfm, "__version__", "unknown"))
print("has_timesfm_25", hasattr(timesfm, "TimesFM_2p5_200M_torch"))
print("autogluon ok")
print("app import ok")
