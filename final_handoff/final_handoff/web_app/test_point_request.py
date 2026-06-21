import json
import urllib.request


payload = json.dumps(
    {"model": "point", "item_id": "apple_fuji_box10kg_high"}
).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1/api/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=240) as response:
    print(response.read().decode("utf-8")[:1200])
