# web-service

Single-page remittance cost estimator.

- `backend/` – minimal synchronous FastAPI app: `GET /metadata` (form options and
  fx-margin defaults) and `POST /predict`.
- `frontend/` – Vite + React + TypeScript page that posts the form values and
  shows the predicted cost % and estimated USD fee.

## 1. Train and copy the model

```bash
cd ../r-models && python train_model.py && cd ../web-service
./sync_model.sh        # copies model.pkl + metadata.json to backend/model/
```

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload      # http://localhost:8000
```

Use the same scikit-learn version for training and serving.

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

The backend URL is set by `VITE_API_URL` in `frontend/.env` (default
`http://localhost:8000`). `npm run build` type-checks and builds for production.
