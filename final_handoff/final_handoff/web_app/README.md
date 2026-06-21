# Forecast Web App

FastAPI wrapper for the two final handoff models.

- `POINT_FORECAST`: TimesFM 2.5 Zero-Shot, 3-day point forecast.
- `PROBABILISTIC_FORECAST`: Chronos2 LoRA baseline, 10-day quantile forecast.

## Run

```bash
cd final_handoff/final_handoff
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r web_app/requirements-web.txt
uvicorn web_app.app:app --host 0.0.0.0 --port 8000
```

Open `http://SERVER_IP:8000`.

The first point forecast request downloads the TimesFM weights from HuggingFace. The first probabilistic request loads the bundled AutoGluon Chronos2 LoRA predictor from `PROBABILISTIC_FORECAST/model`.

## API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/items
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"both","item_id":"apple_fuji_box10kg_high"}'
```

For Chronos2, future known covariates should be connected to external sources in production. Until those feeds are wired, the web app fills unavailable future known covariates with each item's latest observed value.
