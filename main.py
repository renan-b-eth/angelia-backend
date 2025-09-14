# angelia-backend/main.py (versão com Docker e FFMPEG para conversão)
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
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import subprocess # Para usar ffmpeg

load_dotenv()

BASE_DIR = Path(__file__).parent
DATASET_AUDIO_DIR = BASE_DIR / "dataset/audios"
MODEL_PATH = BASE_DIR / "models/modelo_svm.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL não configurada.")

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
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000", "http://localhost:8000"]

class AddToDatasetResponse(BaseModel):
    message: str
    audio_filename: str
    dataset_entry_id: str

app = FastAPI(title="angel.ia Backend API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)

# Função para converter áudio para WAV usando ffmpeg
def convert_audio_to_wav(input_path: Path, output_path: Path):
    try:
        # Verifica se o ffmpeg está disponível
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg não está instalado ou não está no PATH. É necessário para a conversão de áudio.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Erro ao executar ffmpeg: {e.stderr.decode()}")
        
    command = ["ffmpeg", "-i", str(input_path), "-ar", "44100", "-ac", "1", "-y", str(output_path)]
    result = subprocess.run(command, capture_output=True, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro na conversão de áudio com FFmpeg: {result.stderr.decode()}")

def extract_features(audio_path: str):
    # (A função de extração robusta que já tínhamos)
    try:
        sound = parselmouth.Sound(audio_path)
        if sound.duration < 0.5: return None
        f0min, f0max = 75, 600
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)
        jitter_local, shimmer_local = 0.0, 0.0
        try:
            pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            if hasattr(pulses, 'get_number_of_points') and pulses.get_number_of_points() > 0:
                jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer_local = parselmouth.praat.call(pulses, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception: pass
        mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        return {'jitter_local': jitter_local, 'shimmer_local': shimmer_local, 'mean_pitch': mean_pitch, 'mean_hnr': mean_hnr}
    except Exception: return None

@app.post("/add-to-dataset/", response_model=AddToDatasetResponse)
async def add_to_dataset(diagnosis: str = Form(), age: int = Form(), gender: str = Form(), audio_file: UploadFile = File()):
    db = SessionLocal()
    temp_input_path = None
    temp_wav_path = None
    try:
        # Salva o arquivo original temporariamente
        temp_input_path = DATASET_AUDIO_DIR / f"{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Converte o arquivo para WAV
        unique_wav_filename = f"{uuid.uuid4()}.wav"
        temp_wav_path = DATASET_AUDIO_DIR / unique_wav_filename
        convert_audio_to_wav(temp_input_path, temp_wav_path)
        
        features = extract_features(str(temp_wav_path))
        if not features:
            raise HTTPException(400, "Falha ao extrair features do áudio após conversão.")
        
        new_entry = DatasetEntry(audio_filename=unique_wav_filename, diagnosis=diagnosis, age=age, gender=gender, **features)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        
        return AddToDatasetResponse(message="Dados adicionados com sucesso!", audio_filename=unique_wav_filename, dataset_entry_id=new_entry.id)
    except HTTPException: # Re-raise HTTPExceptions diretamente
        raise
    except Exception as e:
        db.rollback()
        # Remova arquivos temporários em caso de erro
        if temp_input_path and temp_input_path.exists(): os.remove(temp_input_path)
        if temp_wav_path and temp_wav_path.exists(): os.remove(temp_wav_path)
        raise HTTPException(500, f"Erro interno: {e}")
    finally:
        db.close()
        # Remova o arquivo original, mas mantenha o WAV convertido se tudo deu certo
        if temp_input_path and temp_input_path.exists():
            os.remove(temp_input_path)