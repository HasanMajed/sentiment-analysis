import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from collections import Counter
from app.preprocessing import clean_text

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Load TF-IDF models and shared vectorizer
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
nb_model = joblib.load(os.path.join(MODEL_DIR, 'multinominalnb_model.pkl'))
lr_model = joblib.load(os.path.join(MODEL_DIR, 'logisticregression_model.pkl'))
svc_model = joblib.load(os.path.join(MODEL_DIR, 'svc_model.pkl'))
sgd_model = joblib.load(os.path.join(MODEL_DIR, 'sgdclassifier_model.pkl'))
knn_model = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))
stacker_model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))

app = FastAPI(
    title="TF-IDF Sentiment API",
    description="Predict sentiment from text using TF-IDF based models",
    docs_url="/docs",
    redoc_url=None
)

class SentimentRequest(BaseModel):
    text: str = Field(..., example="This product is amazing and works like a charm!", description="Text to analyze for sentiment.")

    class Config:
        schema_extra = {
            "example": {
                "text": "I love this app! It's so useful and easy to use."
            }
        }

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., example="positive", description="Predicted sentiment label.")
    confidence: float = Field(..., example=0.9453, description="Model confidence for the prediction.")

    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9453
            }
        }

@app.get("/api/v1/health")
def health_check():
    return {"status": "TF-IDF API is running"}

@app.post("/api/v1/multinominal/predict", response_model=SentimentResponse)
def multinomial_nb_predict(request: SentimentRequest):
    """Predict sentiment using MultinomialNB model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_nb = tfidf_vectorizer.transform([preprocessed_text])
    pred_nb = nb_model.predict(X_nb)[0]
    
    # Get confidence score
    conf_nb = 0.0
    if hasattr(nb_model, "predict_proba"):
        proba_nb = nb_model.predict_proba(X_nb)
        conf_nb = float(proba_nb.max())
    
    return SentimentResponse(sentiment=str(pred_nb), confidence=round(conf_nb, 4))

@app.post("/api/v1/logisticregression/predict", response_model=SentimentResponse)
def logistic_regression_predict(request: SentimentRequest):
    """Predict sentiment using Logistic Regression model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_lr = tfidf_vectorizer.transform([preprocessed_text])
    pred_lr = lr_model.predict(X_lr)[0]
    
    # Get confidence score
    conf_lr = 0.0
    if hasattr(lr_model, "predict_proba"):
        proba_lr = lr_model.predict_proba(X_lr)
        conf_lr = float(proba_lr.max())
    
    return SentimentResponse(sentiment=str(pred_lr), confidence=round(conf_lr, 4))

@app.post("/api/v1/svc/predict", response_model=SentimentResponse)
def svc_predict(request: SentimentRequest):
    """Predict sentiment using Support Vector Classifier model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_svc = tfidf_vectorizer.transform([preprocessed_text])
    pred_svc = svc_model.predict(X_svc)[0]
    
    # Get confidence score
    conf_svc = 0.0
    if hasattr(svc_model, "predict_proba"):
        proba_svc = svc_model.predict_proba(X_svc)
        conf_svc = float(proba_svc.max())
    
    return SentimentResponse(sentiment=str(pred_svc), confidence=round(conf_svc, 4))

@app.post("/api/v1/sgd/predict", response_model=SentimentResponse)
def sgd_predict(request: SentimentRequest):
    """Predict sentiment using SGD Classifier model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_sgd = tfidf_vectorizer.transform([preprocessed_text])
    pred_sgd = sgd_model.predict(X_sgd)[0]
    
    # Get confidence score
    conf_sgd = 0.0
    if hasattr(sgd_model, "predict_proba"):
        proba_sgd = sgd_model.predict_proba(X_sgd)
        conf_sgd = float(proba_sgd.max())
    
    return SentimentResponse(sentiment=str(pred_sgd), confidence=round(conf_sgd, 4))

@app.post("/api/v1/knn/predict", response_model=SentimentResponse)
def knn_predict(request: SentimentRequest):
    """Predict sentiment using K-Nearest Neighbors model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_knn = tfidf_vectorizer.transform([preprocessed_text])
    pred_knn = knn_model.predict(X_knn)[0]
    
    # Get confidence score
    conf_knn = 0.0
    if hasattr(knn_model, "predict_proba"):
        proba_knn = knn_model.predict_proba(X_knn)
        conf_knn = float(proba_knn.max())
    
    return SentimentResponse(sentiment=str(pred_knn), confidence=round(conf_knn, 4))

@app.post("/api/v1/stacker/predict", response_model=SentimentResponse)
def stacker_predict(request: SentimentRequest):
    """Predict sentiment using Stacking Classifier model with TF-IDF vectorizer."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    # Transform text using TF-IDF vectorizer
    X_stacker = tfidf_vectorizer.transform([preprocessed_text])
    pred_stacker = stacker_model.predict(X_stacker)[0]
    
    # Get confidence score
    conf_stacker = 0.0
    if hasattr(stacker_model, "predict_proba"):
        proba_stacker = stacker_model.predict_proba(X_stacker)
        conf_stacker = float(proba_stacker.max())
    
    return SentimentResponse(sentiment=str(pred_stacker), confidence=round(conf_stacker, 4))

@app.post("/api/v1/text-sentiment", response_model=SentimentResponse)
def tfidf_ensemble_predict(request: SentimentRequest):
    """Predict sentiment using all TF-IDF models and return the most confident result."""
    text = request.text.strip()
    preprocessed_text = clean_text(text)
    
    if not preprocessed_text.strip():
        raise HTTPException(status_code=400, detail="Input has no meaningful content after preprocessing.")
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    
    tfidf_preds = []
    
    # MultinomialNB
    X_nb = tfidf_vectorizer.transform([preprocessed_text])
    pred_nb = nb_model.predict(X_nb)[0]
    conf_nb = 0.0
    if hasattr(nb_model, "predict_proba"):
        proba_nb = nb_model.predict_proba(X_nb)
        conf_nb = float(proba_nb.max())
    tfidf_preds.append((str(pred_nb), conf_nb))
    
    # LogisticRegression
    X_lr = tfidf_vectorizer.transform([preprocessed_text])
    pred_lr = lr_model.predict(X_lr)[0]
    conf_lr = 0.0
    if hasattr(lr_model, "predict_proba"):
        proba_lr = lr_model.predict_proba(X_lr)
        conf_lr = float(proba_lr.max())
    tfidf_preds.append((str(pred_lr), conf_lr))
    
    # SVC
    X_svc = tfidf_vectorizer.transform([preprocessed_text])
    pred_svc = svc_model.predict(X_svc)[0]
    conf_svc = 0.0
    if hasattr(svc_model, "predict_proba"):
        proba_svc = svc_model.predict_proba(X_svc)
        conf_svc = float(proba_svc.max())
    tfidf_preds.append((str(pred_svc), conf_svc))
    
    # SGD
    X_sgd = tfidf_vectorizer.transform([preprocessed_text])
    pred_sgd = sgd_model.predict(X_sgd)[0]
    conf_sgd = 0.0
    if hasattr(sgd_model, "predict_proba"):
        proba_sgd = sgd_model.predict_proba(X_sgd)
        conf_sgd = float(proba_sgd.max())
    tfidf_preds.append((str(pred_sgd), conf_sgd))
    
    # KNN
    X_knn = tfidf_vectorizer.transform([preprocessed_text])
    pred_knn = knn_model.predict(X_knn)[0]
    conf_knn = 0.0
    if hasattr(knn_model, "predict_proba"):
        proba_knn = knn_model.predict_proba(X_knn)
        conf_knn = float(proba_knn.max())
    tfidf_preds.append((str(pred_knn), conf_knn))
    
    # Stacker
    X_stacker = tfidf_vectorizer.transform([preprocessed_text])
    pred_stacker = stacker_model.predict(X_stacker)[0]
    conf_stacker = 0.0
    if hasattr(stacker_model, "predict_proba"):
        proba_stacker = stacker_model.predict_proba(X_stacker)
        conf_stacker = float(proba_stacker.max())
    tfidf_preds.append((str(pred_stacker), conf_stacker))
    
    # Find prediction with highest confidence
    best_pred, best_conf = max(tfidf_preds, key=lambda x: x[1])
    
    return SentimentResponse(sentiment=best_pred, confidence=round(best_conf, 4))