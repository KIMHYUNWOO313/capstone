from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django


django.setup()

from dashboard.models import AgriActual, AgriPredictBatch  # noqa: E402


predict_deleted = AgriPredictBatch.objects.all().delete()
actual_deleted = AgriActual.objects.all().delete()

print(f"predict_deleted={predict_deleted}")
print(f"actual_deleted={actual_deleted}")
