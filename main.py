# angelia-backend/main.py (versão COMPLETA e FINAL com Cloudflare R2 e correção de tipos Numpy)
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
import soundfile as sf
import boto3

load_dotenv()

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/modelo_svm.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Variáveis de Ambiente para Cloudflare R2 ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL_BASE = os.getenv("R2_PUBLIC_URL_BASE")

# Validação das variáveis de ambiente
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL não está configurada.")
if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME, R2_PUBLIC_URL_BASE]):
    raise RuntimeError("Variáveis de ambiente R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME, R2_PUBLIC_URL_BASE são obrigatórias para Cloudflare R2.")

# Construir o endpoint de serviço R2 (compatível com S3)
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Inicializar cliente S3 (boto3, mas apontando para R2)
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto' # R2 não usa regiões AWS tradicionais, 'auto' é comum com R2
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DatasetEntry(Base):
    __tablename__ = "dataset_entries"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_url = Column(String, index=True) 
    diagnosis = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    mfcc_mean = Column(Float)
    pitch_mean = Column(Float)
    spectral_centroid_mean = Column(Float)
    zero_crossing_rate_mean = Column(Float)

Base.metadata.create_all(bind=engine)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000"]

class AddToDatasetResponse(BaseModel):
    message: str
    audio_url: str
    dataset_entry_id: str

app = FastAPI(title="angel.ia Backend API", version="5.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo não carregado. Erro: {e}.")

# --- Função para converter áudio para WAV usando ffmpeg ---
def convert_to_wav_with_ffmpeg(input_path: Path, output_path: Path):
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
        y, sr = librosa.load(str(audio_path), sr=None)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs) if mfccs.size > 0 else 0.0

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr)
        pitch_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid) if spectral_centroid.size > 0 else 0.0
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate) if zero_crossing_rate.size > 0 else 0.0

        # --- CORREÇÃO AQUI: Converter numpy types para float nativo Python ---
        return {
            'mfcc_mean': float(mfcc_mean),
            'pitch_mean': float(pitch_mean),
            'spectral_centroid_mean': float(spectral_centroid_mean),
            'zero_crossing_rate_mean': float(zero_crossing_rate_mean)
        }
        # --- FIM DA CORREÇÃO ---
    except Exception as e:
        print(f"[ERRO Librosa] Falha na extração de features: {e}")
        return None

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia com Cloudflare R2!"}

@app.post("/add-to-dataset/", response_model=AddToDatasetResponse)
async def add_to_dataset(diagnosis: str = Form(...), age: int = Form(...), gender: str = Form(...), audio_file: UploadFile = File(...)):
    db = SessionLocal()
    temp_webm_path, temp_wav_path = None, None
    audio_r2_url = None
    try:
        print("[DEBUG] Iniciando processamento do áudio.")
        # 1. Salva o arquivo original (webm) temporariamente no container
        temp_webm_path = BASE_DIR / f"temp_input_{uuid.uuid4()}_{audio_file.filename}"
        with temp_webm_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[DEBUG] WEBM temporário salvo em: {temp_webm_path}")

        # 2. Converte o arquivo temporário (webm) para WAV usando ffmpeg no container
        temp_wav_path = BASE_DIR / f"temp_converted_{uuid.uuid4()}.wav"
        if not convert_to_wav_with_ffmpeg(temp_webm_path, temp_wav_path):
            print("[ERROR] Falha na conversão para WAV.")
            raise HTTPException(400, "Falha ao converter áudio para WAV.")
        print(f"[DEBUG] WAV convertido salvo em: {temp_wav_path}")
        
        # 3. Extrai features do arquivo WAV convertido com Librosa
        features = extract_features_librosa(temp_wav_path)
        if not features:
            print("[ERROR] Falha na extração de features.")
            raise HTTPException(400, "Falha ao extrair features do áudio após conversão.")
        print(f"[DEBUG] Features extraídas: {features}")

        # 4. Envia o arquivo WAV convertido para o R2 (usando cliente S3)
        r2_object_key = f"audios/{uuid.uuid4()}.wav" # Caminho dentro do bucket R2
        print(f"[DEBUG] Tentando upload para R2. Bucket: {R2_BUCKET_NAME}, Key: {r2_object_key}")
        
        s3_client.upload_file(str(temp_wav_path), R2_BUCKET_NAME, r2_object_key)
        
        print(f"[DEBUG] Upload para R2 bem-sucedido para key: {r2_object_key}")
        
        audio_r2_url = f"{R2_PUBLIC_URL_BASE}/{r2_object_key}"
        print(f"[DEBUG] URL pública do R2: {audio_r2_url}")

        # 5. Salva a URL do R2 na base de dados
        new_entry = DatasetEntry(audio_url=audio_r2_url, diagnosis=diagnosis, age=age, gender=gender, **features)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        print(f"[DEBUG] Entrada no banco de dados criada com ID: {new_entry.id}")

        return {"message": "Dados adicionados com sucesso!", "audio_url": audio_r2_url, "dataset_entry_id": new_entry.id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"[ERRO R2/Interno] Falha no processamento: {e}") # Captura qualquer erro
        raise HTTPException(500, f"Erro interno no servidor ao processar o áudio: {e}")
    finally:
        print("[DEBUG] Iniciando limpeza de arquivos temporários.")
        if temp_webm_path and temp_webm_path.exists():
            os.remove(temp_webm_path)
            print(f"[DEBUG] Removido temp_webm_path: {temp_webm_path}")
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)
            print(f"[DEBUG] Removido temp_wav_path: {temp_wav_path}")
        print("[DEBUG] Limpeza de arquivos temporários concluída.")