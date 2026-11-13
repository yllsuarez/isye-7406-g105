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

## Run in VS Code

Open this `web-service/` folder in VS Code (or load it as part of the repo's
multi-root `isye-7406-g105.code-workspace`). After completing steps 1–2's setup
(venv + `pip install -r requirements.txt`, `npm install`), use whichever of the
following suits your workflow:

- **Debug the backend**: install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python),
  select the `backend/.venv` interpreter (Command Palette → "Python: Select
  Interpreter"), then open Run & Debug and launch **"Python: FastAPI
  (uvicorn)"** (defined in `.vscode/launch.json`). This starts `uvicorn
  main:app --reload` with breakpoints enabled in `backend/main.py`.
- **Quick-run without debugging**: open the Command Palette → "Tasks: Run
  Task" and pick **"Run Backend"** or **"Run Frontend"** (defined in
  `.vscode/tasks.json`). Each opens a dedicated terminal panel running the
  same commands as steps 2–3 below.

Both approaches are equivalent to the manual CLI steps in sections 2–3 above —
use whichever is convenient, or fall back to the raw `uvicorn`/`npm run dev`
commands if you're not using VS Code.
