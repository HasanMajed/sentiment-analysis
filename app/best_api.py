import os
import re
import csv
import joblib
import numpy as np
import pandas as pd

from io import StringIO
from collections import Counter
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from app.preprocessing import clean_text, mark_negation

# --- Utility functions ---
def normalize_apostrophes(text: str) -> str:
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"\bdont\b", "don't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoesnt\b", "doesn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdidnt\b", "didn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwasnt\b", "wasn't", text, flags=re.IGNORECASE)
    return text

NEUTRAL_PATTERNS = re.compile(
    r"^(?:it's?|its|it is)\s+(?:ok|okay|fine|so[-\s]?so|meh)$"
    r"|^meh$|^whatever$|^fine$",
    re.IGNORECASE
)
def detect_neutral(text: str) -> bool:
    return bool(NEUTRAL_PATTERNS.match(text.strip()))

def is_complex(text: str) -> bool:
    if len(text.split()) > 10:
        return True
    if re.search(r"\bbut\b|\bhowever\b|\balthough\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\bdo(?:n't| not)\s+\w+", text, flags=re.IGNORECASE):
        return True
    return False

# --- Load models & tokenizers ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, 'models')

tfidf_vectorizer = joblib.load(os.path.join(MDIR, 'tfidf_vectorizer.pkl'))
count_vectorizer = joblib.load(os.path.join(MDIR, 'count_vectorizer.pkl'))
label_encoder    = joblib.load(os.path.join(MDIR, 'label_encoder.pkl'))

models_tfidf = {
    "MultinomialNB": joblib.load(os.path.join(MDIR, 'multinominalnb_model.pkl')),
    "LogisticRegression": joblib.load(os.path.join(MDIR, 'logisticregression_model.pkl')),
    "SVC": joblib.load(os.path.join(MDIR, 'svc_model.pkl')),
    "SGD": joblib.load(os.path.join(MDIR, 'sgdclassifier_model.pkl')),
    "KNN": joblib.load(os.path.join(MDIR, 'knn_model.pkl')),
    "Stacker": joblib.load(os.path.join(MDIR, 'stacking_model.pkl')),
}

models_ensemble = {
    "RandomForest": joblib.load(os.path.join(MDIR, 'random_forest_model.pkl')),
    "XGBoost":      joblib.load(os.path.join(MDIR, 'xgboost_model.pkl')),
    "CatBoost":     joblib.load(os.path.join(MDIR, 'catboost_model.pkl')),
    "LightGBM":     joblib.load(os.path.join(MDIR, 'lightgbm_model.pkl')),
}

cnn_model  = load_model(os.path.join(MDIR, 'sentiment_cnn_model.h5'), compile=False)
lstm_model = load_model(os.path.join(MDIR, 'sentiment_lstm_model.h5'), compile=False)

with open(os.path.join(MDIR, 'tokenizer.json'), 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())

MAX_SEQ_LEN = 60

# --- FastAPI setup ---
app = FastAPI(docs_url="/docs", redoc_url=None)

class SentimentRequest(BaseModel):
    text: str = Field(..., example="I don't love it")

def analyze_text(raw_text: str) -> dict:
    raw = normalize_apostrophes(raw_text or "")
    if not raw.strip():
        return {"text": raw_text, "sentiment": "neutral", "confidence": 0.0}

    neutral_flag = detect_neutral(raw)
    processed = mark_negation(clean_text(raw))
    if not processed.strip():
        return {"text": raw_text, "sentiment": "neutral", "confidence": 0.0}

    outputs = []
    complex_flag = is_complex(raw)

    # classical ML (skip if complex)
    if not complex_flag:
        X_tf = tfidf_vectorizer.transform([processed])
        for name, m in models_tfidf.items():
            p = m.predict(X_tf)[0]
            c = float(m.predict_proba(X_tf).max()) if hasattr(m, "predict_proba") else 0.0
            outputs.append((name, p, c))

        X_ct = count_vectorizer.transform([processed]).toarray().astype(np.float32)
        for name, m in models_ensemble.items():
            idx = m.predict(X_ct)[0]
            lbl = label_encoder.inverse_transform([idx])[0]
            c   = float(m.predict_proba(X_ct)[0].max())
            outputs.append((name, lbl, c))

    # deep models always run
    seq = tokenizer.texts_to_sequences([processed])[0]
    vocab = next(layer for layer in cnn_model.layers if isinstance(layer, Embedding)).input_dim
    seq = [t if t < vocab else 1 for t in seq]
    pad = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    for name, m in [("CNN", cnn_model), ("LSTM", lstm_model)]:
        probs = m.predict(pad, verbose=0)[0]
        idx   = int(np.argmax(probs))
        c     = float(np.max(probs))
        lbl   = label_encoder.inverse_transform([idx])[0]
        outputs.append((name, lbl, c))

    # voting set
    vote_set = [o for o in outputs if o[0] in ("CNN", "LSTM")] if complex_flag else outputs

    votes = [o[1] for o in vote_set]
    cnts  = Counter(votes)
    top, topc = cnts.most_common(1)[0]
    ties = [lbl for lbl, ct in cnts.items() if ct == topc]

    if neutral_flag or len(ties) > 1:
        final_sent = "neutral"
        final_conf = float(np.mean([o[2] for o in vote_set]))
    else:
        final_sent = top
        final_conf = float(np.mean([o[2] for o in vote_set if o[1] == final_sent]))

    return {"text": raw_text, "sentiment": final_sent, "confidence": round(final_conf, 4)}

@app.get("/api/v1/health")
def health_check():
    return {"status": "API is running"}

@app.post("/api/v1/text-analysis")
def text_analysis(request: SentimentRequest):
    return analyze_text(request.text)

@app.post("/api/v1/batch-analysis")
async def batch_analysis(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    data = (await file.read()).decode("utf-8")
    if ext == "csv":
        df = pd.read_csv(StringIO(data))
    elif ext == "json":
        df = pd.read_json(StringIO(data))
    else:
        raise HTTPException(status_code=400, detail="Must upload CSV or JSON")

    first_col = df.columns[0]
    results = []
    for txt in df[first_col].fillna("").astype(str):
        res = analyze_text(txt)
        results.append({
            first_col: txt,
            "sentiment": res["sentiment"],
            "confidence": res["confidence"]
        })

    out_df = pd.DataFrame(results)
    stream = StringIO()
    out_df.to_csv(
        stream,
        index=False,
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_ALL
    )
    stream.seek(0)

    base, extension = os.path.splitext(file.filename)
    out_name = f"{base}_analyzed{extension}"

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )