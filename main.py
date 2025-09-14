# angelia-backend/main.py (versão FINAL com Librosa e FFMPEG para conversão)
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
import soundfile as sf # Importar soundfile

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

app = FastAPI(title="angel.ia Backend API", version="4.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo não carregado. Erro: {e}.")

# --- Função para converter áudio para WAV usando ffmpeg ---
def convert_to_wav_with_ffmpeg(input_path: Path, output_path: Path):
    # -y sobrescreve o arquivo de saída sem perguntar
    command = ["ffmpeg", "-i", str(input_path), "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", "-y", str(output_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[Conversão] Concluído: {input_path.name} -> {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO FFmpeg] Falha na conversão. Stderr: {e.stderr}")
        return False

# --- Função de Extração de Features com Librosa ---
def extract_features_librosa(audio_path: Path) -> dict | None:
    try:
        # Librosa carrega do arquivo WAV já no disco
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs) if mfccs.size > 0 else 0.0

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr)
        pitch_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0 # Ignorar silêncio
        
        # Centroide Espectral
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid) if spectral_centroid.size > 0 else 0.0
        
        # Taxa de Cruzamento por Zero
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate) if zero_crossing_rate.size > 0 else 0.0

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
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia"}

@app.post("/add-to-dataset/")
async def add_to_dataset(diagnosis: str = Form(...), age: int = Form(...), gender: str = Form(...), audio_file: UploadFile = File(...)):
    db = SessionLocal()
    temp_webm_path, temp_wav_path = None, None
    try:
        # Salva o arquivo original (webm) temporariamente
        temp_webm_path = BASE_DIR / f"temp_input_{uuid.uuid4()}_{audio_file.filename}"
        with temp_webm_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Converte o arquivo temporário (webm) para WAV usando ffmpeg
        temp_wav_path = BASE_DIR / f"temp_converted_{uuid.uuid4()}.wav"
        if not convert_to_wav_with_ffmpeg(temp_webm_path, temp_wav_path):
            raise HTTPException(400, "Falha ao converter áudio para WAV.")
        
        # Extrai features do arquivo WAV convertido com Librosa
        features = extract_features_librosa(temp_wav_path)
        if not features:
            raise HTTPException(400, "Falha ao extrair features do áudio após conversão.")

        # Salva o arquivo WAV convertido permanentemente
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_audio_filename
        shutil.move(temp_wav_path, permanent_audio_path)
        temp_wav_path = None # Limpa a referência para não tentar apagar arquivo movido

        new_entry = DatasetEntry(audio_filename=unique_audio_filename, diagnosis=diagnosis, age=age, gender=gender, **features)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        return {"message": "Dados adicionados com sucesso!", "audio_filename": unique_audio_filename, "dataset_entry_id": new_entry.id}
    except HTTPException: # Re-raise HTTPExceptions diretamente
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Erro interno no servidor: {e}")
    finally:
        # Garante que os arquivos temporários sejam sempre limpos
        if temp_webm_path and temp_webm_path.exists():
            os.remove(temp_webm_path)
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)