# angelia-backend/main.py (versão com Librosa)
import os
import shutil
import pandas as pd
import joblib
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import librosa
import numpy as np
import io

load_dotenv()

BASE_DIR = Path(__file__).parent
DATASET_AUDIO_DIR = BASE_DIR / "dataset/audios"
MODEL_PATH = BASE_DIR / "models/modelo_svm.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL não está configurada.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DatasetEntry(Base):
    __tablename__ = "dataset_entries"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_filename = Column(String, index=True)
    diagnosis = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # Features do Librosa
    mfcc_mean = Column(Float)
    pitch_mean = Column(Float)
    spectral_centroid_mean = Column(Float)
    zero_crossing_rate_mean = Column(Float)

Base.metadata.create_all(bind=engine)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000"]

class AddToDatasetResponse(BaseModel):
    message: str
    audio_filename: str
    dataset_entry_id: str

app = FastAPI(title="angel.ia Backend API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)

# --- Função de Extração de Features com Librosa ---
def extract_features_librosa(audio_bytes: bytes) -> dict | None:
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # MFCCs (coeficientes cepstrais de frequência mel)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs)

        # Pitch (Frequência Fundamental)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0.0

        # Centroide Espectral
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # Taxa de Cruzamento por Zero
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)

        return {
            'mfcc_mean': mfcc_mean,
            'pitch_mean': pitch_mean,
            'spectral_centroid_mean': spectral_centroid_mean,
            'zero_crossing_rate_mean': zero_crossing_rate_mean
        }
    except Exception as e:
        print(f"[ERRO Librosa] Falha na extração de features: {e}")
        return None

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/add-to-dataset/")
async def add_to_dataset(diagnosis: str = Form(...), age: int = Form(...), gender: str = Form(...), audio_file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        audio_bytes = await audio_file.read()
        
        features = extract_features_librosa(audio_bytes)
        if not features:
            raise HTTPException(400, "Falha ao extrair features do áudio.")

        unique_filename = f"{uuid.uuid4()}.{audio_file.filename.split('.')[-1]}"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_filename
        with permanent_audio_path.open("wb") as buffer:
            buffer.write(audio_bytes)

        new_entry = DatasetEntry(audio_filename=unique_filename, diagnosis=diagnosis, age=age, gender=gender, **features)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        return {"message": "Dados adicionados com sucesso!", "audio_filename": unique_filename, "dataset_entry_id": new_entry.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Erro interno: {e}")
    finally:
        db.close()