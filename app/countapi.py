from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from collections import Counter
from app.preprocessing import clean_text

app = FastAPI(
    title="Text Sentiment API",
    version="3.0",
    description="Predict sentiment from text using an ensemble of 6 models with majority voting",
    docs_url="/docs",
    redoc_url=None
)

vectorizer = joblib.load("models/count_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

models = {
    "random_forest":    joblib.load("models/random_forest_model.pkl"),
    "xgboost":          joblib.load("models/xgboost_model.pkl"),
    "catboost":         joblib.load("models/catboost_model.pkl"),
    "gaussian_nb":      joblib.load("models/gaussian_nb_model.pkl"),
    "lightgbm":         joblib.load("models/lightgbm_model.pkl"),
    "decision_tree":    joblib.load("models/decision_tree_model.pkl"),
}

class SentimentRequest(BaseModel):
    text: str = Field(..., example="This product is amazing and works like a charm!")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., example="positive")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.9453)

@app.get("/api/v1/health")
def health_check():
    return {"status": "API is running"}

@app.post("/api/v1/sentiment-predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    text = request.text.strip()
    preprocessed_text = clean_text(text)

    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")

    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned]).toarray().astype(np.float32)

    preds, probs = [], []
    for model in models.values():
        idx = model.predict(X)[0]
        label = label_encoder.inverse_transform([idx])[0]
        preds.append(label)
        probs.append(float(np.max(model.predict_proba(X)[0])))

    vote = Counter(preds).most_common(1)[0][0]
    confidence = round(float(np.mean([p for pl, p in zip(preds, probs) if pl == vote])), 4)

    return SentimentResponse(sentiment=vote, confidence=confidence)