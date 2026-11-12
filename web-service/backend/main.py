"""Minimal synchronous prediction backend: GET /metadata and POST /predict."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent / "model"
pipeline = joblib.load(MODEL_DIR / "model.pkl")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = [f["name"] for f in metadata["features"]]
CATEGORIES = metadata["categories"]

app = FastAPI(title="Remittance cost predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    destination_name: str
    firm_type_clean: str
    payment_instrument_group: str
    access_point: str
    receiving_network_coverage: str
    speed_actual: str
    pickup_method: str
    cc1_fx_margin: float
    year: int = Field(ge=2000, le=2100)
    is_non_transparent: bool = False
    transfer_amount: float | None = Field(default=None, ge=0)


class PredictResponse(BaseModel):
    predicted_cost_pct: float
    estimated_fee_usd: float | None


@app.get("/metadata")
def get_metadata() -> dict:
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    values = {
        "destination_name": req.destination_name,
        "firm_type_clean": req.firm_type_clean,
        "payment_instrument_group": req.payment_instrument_group,
        "access point": req.access_point,
        "receiving network coverage": req.receiving_network_coverage,
        "speed actual": req.speed_actual.strip().lower(),
        "pickup method": req.pickup_method.strip().lower(),
        "cc1 fx margin": req.cc1_fx_margin,
        "year": req.year,
        "is_non_transparent": int(req.is_non_transparent),
    }
    for col, options in CATEGORIES.items():
        if values[col] not in options:
            raise HTTPException(
                status_code=422, detail=f"Unknown value {values[col]!r} for '{col}'"
            )
    row = pd.DataFrame([values], columns=FEATURES)
    pct = float(pipeline.predict(row)[0])
    fee = pct / 100 * req.transfer_amount if req.transfer_amount is not None else None
    return PredictResponse(
        predicted_cost_pct=round(pct, 4),
        estimated_fee_usd=round(fee, 2) if fee is not None else None,
    )
