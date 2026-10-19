# web-service

Single-page Streamlit app that loads the trained model, takes user input,
and displays the prediction.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the exported model file (`model.pkl`, from `../r-models` or a Python
training script) in this folder.

## Run locally

```bash
streamlit run app.py
```

## Deploy

Push to GitHub and connect the repo on
[Streamlit Community Cloud](https://share.streamlit.io), pointing it at
`web-service/app.py`.
