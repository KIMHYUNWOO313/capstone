import inspect

import timesfm


print(timesfm.__file__)
print([name for name in dir(timesfm) if "Times" in name or "Forecast" in name or "Config" in name])
for name in dir(timesfm):
    if name.startswith("Times") or name.endswith("Config"):
        obj = getattr(timesfm, name)
        try:
            print(name, inspect.signature(obj))
        except Exception as exc:
            print(name, type(obj), exc)

print("forecast signature", inspect.signature(timesfm.TimesFm.forecast))
print("init source")
print(inspect.getsource(timesfm.TimesFm.__init__))
print("load source")
print(inspect.getsource(timesfm.TimesFm.load_from_checkpoint))
