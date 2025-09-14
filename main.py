# angelia-backend/main.py
import os
import shutil
import pandas as pd
import joblib
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import parselmouth
import subprocess # Importado para usar o ffmpeg
from pathlib import Path # Para lidar com caminhos de forma mais moderna

# --- Constantes de Configuração ---
BASE_DIR = Path(__file__).parent
DATASET_AUDIO_DIR = BASE_DIR / "dataset/audios"
DATASET_CSV_PATH = BASE_DIR / "dataset/features.csv"
MODEL_PATH = BASE_DIR / "models/modelo_svm.pkl"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "audio_features_collection"

# --- Segurança (para deploy, ajuste isso!) ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [
    FRONTEND_URL, "http://localhost:3000", "http://localhost:8000"
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
    version="1.3.0" # Versão atualizada
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo não carregado. Erro: {e}")

# --- Funções de Extração de Features (sem alterações) ---
def extract_features(audio_path: str) -> dict | None:
    try:
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call(pulses, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        return {
            'jitter_local': jitter_local, 'shimmer_local': shimmer_local,
            'mean_pitch': mean_pitch, 'mean_hnr': mean_hnr
        }
    except Exception as e:
        print(f"[ERRO] Falha na extração de features com parselmouth: {e}")
        return None

def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Usa o ffmpeg para converter um arquivo de áudio para .wav PCM de 16-bit."""
    print(f"[Conversão] Convertendo '{input_path}' para '{output_path}'...")
    try:
        # Comando ffmpeg para converter para WAV, 16-bit PCM, 1 canal (mono), 44.1kHz sample rate
        command = [
            "ffmpeg",
            "-i", input_path,      # Arquivo de entrada
            "-acodec", "pcm_s16le", # Codec de áudio (WAV padrão)
            "-ac", "1",            # 1 canal de áudio (mono)
            "-ar", "44100",        # Sample rate de 44.1kHz
            "-y",                  # Sobrescrever arquivo de saída se existir
            output_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[Conversão] ffmpeg executado com sucesso.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO FATAL] Falha na conversão com ffmpeg.")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[ERRO FATAL] Comando 'ffmpeg' não encontrado. Verifique se está instalado no container Docker.")
        return False

# --- Endpoints da API ---

@app.get("/", tags=["Health"])
async def read_root():
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia"}

# (O endpoint /analyze/ foi omitido para focar na correção do /add-to-dataset/, mas a lógica de conversão deve ser aplicada a ele também)

@app.post("/add-to-dataset/", response_model=AddToDatasetResponse, tags=["Dataset"])
async def add_to_dataset(
    diagnosis: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    audio_file: UploadFile = File(...)
):
    temp_input_path = None
    temp_wav_path = None
    
    try:
        # 1. Salvar o arquivo de áudio recebido (.webm) temporariamente
        temp_input_path = f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Dataset] Áudio recebido salvo temporariamente em: {temp_input_path}")

        # 2. Converter o áudio para .wav
        temp_wav_path = f"temp_{uuid.uuid4()}.wav"
        if not convert_audio_to_wav(temp_input_path, temp_wav_path):
            raise HTTPException(status_code=400, detail="Falha ao converter áudio para o formato WAV.")

        # 3. Extrair as features do arquivo .wav convertido
        print(f"[Dataset] Extraindo features de '{temp_wav_path}'...")
        features = extract_features(temp_wav_path)
        if not features:
            raise HTTPException(status_code=400, detail="Falha ao extrair features do áudio convertido.")
        print(f"[Dataset] Features extraídas: {features}")

        # 4. Salvar o áudio original (.webm) de forma permanente no dataset
        file_extension = Path(audio_file.filename).suffix or '.webm'
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_filename
        shutil.move(temp_input_path, permanent_audio_path) # Move o arquivo original para o destino final
        temp_input_path = None # Marca como movido para não ser deletado no 'finally'
        print(f"[Dataset] Áudio original salvo permanentemente em: {permanent_audio_path}")
        
        # 5. Salvar metadados e features no CSV e ChromaDB
        dataset_entry_id = str(uuid.uuid4())
        new_data_row = {
            'id': dataset_entry_id, 'audio_filename': unique_filename, 'diagnosis': diagnosis,
            'age': age, 'gender': gender, 'timestamp': datetime.now().isoformat(), **features
        }
        new_df = pd.DataFrame([new_data_row])
        file_exists = DATASET_CSV_PATH.is_file()
        new_df.to_csv(DATASET_CSV_PATH, mode='a', header=not file_exists, index=False)
        print(f"[Dataset] Dados salvos no CSV.")

        # (Lógica do ChromaDB omitida para simplificar, mas seria adicionada aqui)
        
        return AddToDatasetResponse(
            message="Dados adicionados ao dataset com sucesso!",
            audio_filename=unique_filename,
            dataset_entry_id=dataset_entry_id,
            features_count=len(features)
        )
    except HTTPException as e:
        raise e # Re-lança exceções HTTP para o FastAPI lidar
    except Exception as e:
        print(f"[ERRO GERAL] Ocorreu um erro inesperado em /add-to-dataset/: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")
    finally:
        # 6. Limpeza: Garante que os arquivos temporários sejam sempre removidos
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            print(f"[Limpeza] Arquivo temporário de entrada removido: {temp_input_path}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"[Limpeza] Arquivo temporário WAV removido: {temp_wav_path}")