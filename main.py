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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv # Para carregar variáveis de ambiente de um .env local

# Carrega variáveis de ambiente do arquivo .env (apenas para desenvolvimento local)
load_dotenv()

# --- Constantes de Configuração ---
BASE_DIR = Path(__file__).parent
DATASET_AUDIO_DIR = BASE_DIR / "dataset/audios"
MODEL_PATH = BASE_DIR / "models/modelo_svm.pkl"

# --- Configuração do Banco de Dados PostgreSQL ---
# A Render injetará DATABASE_URL automaticamente.
# Para desenvolvimento local, defina DATABASE_URL no seu .env
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL não configurada. Defina no .env ou nas variáveis de ambiente da Render.")

# Ajustar a URL para SQLAlchemy, se necessário (Render geralmente já fornece no formato correto)
# Ex: "postgresql://user:password@host:port/database"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Modelo do Banco de Dados (Tabela) ---
class DatasetEntry(Base):
    __tablename__ = "dataset_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_filename = Column(String, index=True) # Nome do arquivo de áudio
    # NOTA: O arquivo de áudio ainda está sendo salvo localmente no container
    # Para persistência real, ele deveria ir para um Object Storage (S3/Spaces)
    # e aqui seria guardada a URL desse armazenamento.
    diagnosis = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    jitter_local = Column(Float, nullable=True)
    shimmer_local = Column(Float, nullable=True)
    mean_pitch = Column(Float, nullable=True)
    mean_hnr = Column(Float, nullable=True)

# Cria as tabelas no banco de dados se elas não existirem
Base.metadata.create_all(bind=engine)

# --- Segurança (para deploy, ajuste isso!) ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:8000"
]

# --- Estruturas de Resposta ---
class AnalysisReport(BaseModel):
    riskLevel: str
    confidence: float
    recommendation: str
    biomarkers: dict

class AddToDatasetResponse(BaseModel):
    message: str
    audio_filename: str
    dataset_entry_id: str
    features_count: int

# --- Inicialização da API ---
app = FastAPI(
    title="angel.ia Backend API",
    description="API para análise de voz e coleta de dados.",
    version="1.5.0" # Versão atualizada
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado com sucesso.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo '{MODEL_PATH}' não carregado. Erro: {e}. A funcionalidade de análise pode estar limitada.")

# --- Funções de Extração de Features ---
def extract_features(audio_path: str) -> dict | None:
    """
    Extrai features fonéticas de um arquivo de áudio usando Parselmouth.
    Retorna um dicionário de features ou None em caso de falha.
    """
    try:
        sound = parselmouth.Sound(audio_path)

        MIN_DURATION_FOR_COMPLEX_FEATURES = 0.5 # segundos
        if sound.duration < MIN_DURATION_FOR_COMPLEX_FEATURES:
            print(f"[ERRO - Parselmouth] Áudio muito curto ({sound.duration:.2f}s) para extrair features complexas.")
            return None

        f0min, f0max = 75, 600

        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)
        
        jitter_local = 0.0
        shimmer_local = 0.0
        try:
            pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            # Verifica se 'pulses' tem o método e se há pontos antes de tentar calcular
            if hasattr(pulses, 'get_number_of_points') and pulses.get_number_of_points() > 0:
                jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer_local = parselmouth.praat.call(pulses, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            else:
                print("[AVISO - Parselmouth] 'To PointProcess' não detectou ciclos de voz periódicos ou o objeto 'pulses' não tem get_number_of_points. Jitter/Shimmer serão 0.")
        except Exception as e:
            print(f"[AVISO - Parselmouth] Falha ao calcular Jitter/Shimmer: {e}. Serão definidos como 0.")

        mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        
        return {
            'jitter_local': jitter_local, 'shimmer_local': shimmer_local,
            'mean_pitch': mean_pitch, 'mean_hnr': mean_hnr
        }
    except Exception as e:
        print(f"[ERRO FATAL - Parselmouth] Falha geral na extração de features: {e}")
        return None

def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
    """
    Usa o ffmpeg para converter um arquivo de áudio (e.g., .webm) para .wav PCM de 16-bit.
    """
    print(f"[Conversão] Tentando converter '{input_path.name}' para WAV...")
    try:
        command = [
            "ffmpeg",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "44100",
            "-y",
            str(output_path)
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[Conversão] Conversão para WAV concluída com sucesso: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO FATAL - FFmpeg] Falha na conversão para WAV para {input_path.name}.")
        print(f"    ffmpeg stdout: {e.stdout.strip()}")
        print(f"    ffmpeg stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("[ERRO FATAL - FFmpeg] Comando 'ffmpeg' não encontrado. Verifique a instalação no Dockerfile.")
        return False
    except Exception as e:
        print(f"[ERRO GERAL - Conversão] Erro inesperado ao converter {input_path.name}: {e}")
        return False

# --- Endpoints da API ---

@app.get("/", tags=["Health"])
async def read_root():
    """Endpoint de saúde para verificar se a API está funcionando."""
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia"}

@app.post("/analyze/", response_model=AnalysisReport, tags=["Análise"])
async def analyze_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint para analisar um arquivo de áudio e retornar um relatório.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modelo de análise não carregado. Tente novamente mais tarde.")

    temp_input_path = None
    temp_wav_path = None

    try:
        temp_input_path = BASE_DIR / f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Análise] Áudio recebido salvo temporariamente em: {temp_input_path.name}")

        final_audio_path = temp_input_path
        if audio_file.content_type != "audio/wav":
            temp_wav_path = BASE_DIR / f"temp_{uuid.uuid4()}.wav"
            if not convert_audio_to_wav(temp_input_path, temp_wav_path):
                raise HTTPException(status_code=400, detail="Falha ao converter áudio para o formato WAV necessário.")
            final_audio_path = temp_wav_path
        else:
            print("[Análise] Áudio já é WAV, pulando conversão.")

        print(f"[Análise] Extraindo features de '{final_audio_path.name}'...")
        features = extract_features(str(final_audio_path))
        if features is None:
            raise HTTPException(status_code=400, detail="Não foi possível extrair features do áudio fornecido. Garanta que o áudio contém fala clara e com duração suficiente.")
        print(f"[Análise] Features extraídas: {features}")

        features_df = pd.DataFrame([features])
        
        prediction = model.predict(features_df)[0]
        risk_level = "Baixo" if prediction == 0 else "Alto"
        recommendation = "Nenhum risco detectado. Mantenha os exames de rotina." if prediction == 0 else "Risco detectado. Recomendamos procurar um especialista para avaliação."
        
        confidence_score = 0.5
        try:
            proba = model.predict_proba(features_df)[0]
            # Supondo que a classe 1 seja "alto risco" e 0 seja "baixo risco"
            confidence_score = proba[1] if prediction == 1 else proba[0]
            
        except AttributeError:
            print("[Análise] Modelo não suporta predict_proba para cálculo de confiança.")
            pass
        
        return AnalysisReport(
            riskLevel=risk_level,
            confidence=confidence_score,
            recommendation=recommendation,
            biomarkers=features
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERRO GERAL] Ocorreu um erro inesperado em /analyze/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e}")
    finally:
        if temp_input_path and temp_input_path.exists():
            os.remove(temp_input_path)
            print(f"[Limpeza] Temp input removido: {temp_input_path.name}")
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)
            print(f"[Limpeza] Temp WAV removido: {temp_wav_path.name}")


@app.post("/add-to-dataset/", response_model=AddToDatasetResponse, tags=["Dataset"])
async def add_to_dataset(
    diagnosis: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Endpoint para adicionar um arquivo de áudio e seus metadados ao dataset.
    Processa o áudio, extrai features e salva no banco de dados.
    """
    temp_input_path = None
    temp_wav_path = None
    db = SessionLocal() # Abre uma nova sessão de banco de dados
    
    try:
        # 1. Salvar o arquivo de áudio recebido temporariamente
        temp_input_path = BASE_DIR / f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Dataset] Áudio recebido salvo temporariamente em: {temp_input_path.name}")

        # 2. Converter o áudio para .wav para extração de features
        temp_wav_path = BASE_DIR / f"temp_{uuid.uuid4()}.wav"
        if not convert_audio_to_wav(temp_input_path, temp_wav_path):
            raise HTTPException(status_code=400, detail="Falha ao converter áudio para o formato WAV necessário.")
        
        # 3. Extrair as features do arquivo .wav convertido
        print(f"[Dataset] Extraindo features de '{temp_wav_path.name}'...")
        features = extract_features(str(temp_wav_path))
        if features is None:
            raise HTTPException(status_code=400, detail="Falha ao extrair features do áudio convertido. Garanta que o áudio contém fala clara e com duração suficiente.")
        print(f"[Dataset] Features extraídas: {features}")

        # 4. Salvar o áudio original (e.g., .webm) no disco efêmero do container
        # ATENÇÃO: ESTE ARMAZENAMENTO NÃO É PERSISTENTE!
        # Para persistência real do áudio, use um Object Storage (e.g., DigitalOcean Spaces, AWS S3).
        # A URL do Object Storage seria então salva no banco de dados.
        file_extension = Path(audio_file.filename).suffix or '.webm'
        unique_audio_filename = f"{uuid.uuid4()}{file_extension}"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_audio_filename
        
        shutil.move(temp_input_path, permanent_audio_path)
        temp_input_path = None # Marca como movido para não ser deletado no 'finally'
        print(f"[Dataset] Áudio original salvo **TEMPORARIAMENTE** em: {permanent_audio_path.name}")
        print("[Dataset] Lembre-se: Para armazenamento persistente de áudios, use um Object Storage como DigitalOcean Spaces ou AWS S3.")
        
        # 5. Criar uma nova entrada no banco de dados
        new_entry = DatasetEntry(
            audio_filename=unique_audio_filename,
            diagnosis=diagnosis,
            age=age,
            gender=gender,
            jitter_local=features.get('jitter_local'),
            shimmer_local=features.get('shimmer_local'),
            mean_pitch=features.get('mean_pitch'),
            mean_hnr=features.get('mean_hnr')
        )
        db.add(new_entry)
        db.commit() # Salva a entrada no banco de dados
        db.refresh(new_entry) # Atualiza o objeto para ter o ID gerado pelo DB
        print(f"[Dataset] Dados e features salvos no PostgreSQL. ID: {new_entry.id}")
        
        return AddToDatasetResponse(
            message="Dados adicionados ao dataset com sucesso!",
            audio_filename=unique_audio_filename,
            dataset_entry_id=new_entry.id,
            features_count=len(features)
        )
    except HTTPException as e:
        db.rollback() # Em caso de erro, desfaz quaisquer mudanças pendentes no DB
        raise e
    except Exception as e:
        db.rollback()
        print(f"[ERRO GERAL] Ocorreu um erro inesperado em /add-to-dataset/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e}")
    finally:
        db.close() # Fecha a sessão do banco de dados
        if temp_input_path and temp_input_path.exists():
            os.remove(temp_input_path)
            print(f"[Limpeza] Arquivo temporário de entrada removido: {temp_input_path.name}")
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)
            print(f"[Limpeza] Arquivo temporário WAV removido: {temp_wav_path.name}")