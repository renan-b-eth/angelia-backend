# angelia-backend/main.py
import os
import shutil
import pandas as pd
import joblib
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import parselmouth
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

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
    jitter_local = Column(Float)
    shimmer_local = Column(Float)
    mean_pitch = Column(Float)
    mean_hnr = Column(Float)

Base.metadata.create_all(bind=engine)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000"]

app = FastAPI(title="angel.ia Backend API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo não carregado. Erro: {e}.")

def convert_audio_to_wav(input_path: Path, output_path: Path):
    command = ["ffmpeg", "-i", str(input_path), "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", "-y", str(output_path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[Conversão] Conversão para WAV concluída: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO FFmpeg] Falha na conversão. Stderr: {e.stderr}")
        return False

def extract_features(audio_path: str):
    try:
        sound = parselmouth.Sound(audio_path)
        if sound.duration < 0.5: return None
        f0min, f0max = 75, 600
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)
        jitter_local, shimmer_local = 0.0, 0.0
        try:
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = parselmouth.praat.call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception as e:
            print(f"[AVISO Parselmouth] Falha ao calcular Jitter/Shimmer: {e}. Serão 0.")
        mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        return {'jitter_local': jitter_local, 'shimmer_local': shimmer_local, 'mean_pitch': mean_pitch, 'mean_hnr': mean_hnr}
    except Exception as e:
        print(f"[ERRO Parselmouth] Falha geral na extração: {e}")
        return None

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia"}

@app.post("/add-to-dataset/")
async def add_to_dataset(diagnosis: str = Form(...), age: int = Form(...), gender: str = Form(...), audio_file: UploadFile = File(...)):
    db = SessionLocal()
    temp_input_path, temp_wav_path = None, None
    try:
        temp_input_path = BASE_DIR / f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        temp_wav_path = BASE_DIR / f"temp_{uuid.uuid4()}.wav"
        if not convert_audio_to_wav(temp_input_path, temp_wav_path):
            raise HTTPException(400, "Falha ao converter áudio para WAV.")
        
        features = extract_features(str(temp_wav_path))
        if not features:
            raise HTTPException(400, "Falha ao extrair features do áudio.")

        # ATENÇÃO: O áudio original (webm) é salvo de forma temporária.
        # Para persistência real, use um serviço de Object Storage (S3, etc.).
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_audio_filename
        shutil.move(temp_wav_path, permanent_audio_path)
        temp_wav_path = None

        new_entry = DatasetEntry(audio_filename=unique_audio_filename, diagnosis=diagnosis, age=age, gender=gender, **features)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        return {"message": "Dados adicionados com sucesso!", "audio_filename": unique_audio_filename, "dataset_entry_id": new_entry.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Erro interno no servidor: {e}")
    finally:
        db.close()
        if temp_input_path and temp_input_path.exists(): os.remove(temp_input_path)
        if temp_wav_path and temp_wav_path.exists(): os.remove(temp_wav_path)