# angelia-backend/main.py
import os
import shutil
import pandas as pd
import joblib
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware # Para permitir acesso do frontend
from pydantic import BaseModel
import parselmouth
import chromadb
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# --- Constantes de Configuração ---
DATASET_AUDIO_DIR = "dataset/audios"
DATASET_CSV_PATH = "dataset/features.csv" # O CSV ainda é mantido para fácil inspeção
MODEL_PATH = "models/modelo_svm.pkl"
CHROMA_DB_PATH = "chroma_db" # Pasta para o ChromaDB
CHROMA_COLLECTION_NAME = "audio_features_collection"

# --- Segurança (para deploy, ajuste isso!) ---
# Em desenvolvimento, '*' é ok. Em produção, use a URL do seu frontend.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000") 
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000", # Para desenvolvimento local do frontend
    "http://localhost:8000"  # Para Swagger UI ou testes diretos
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
    description="API para análise de voz e coleta de dados para o dataset vetorial.",
    version="1.2.0"
)

# Configura CORS para permitir que o frontend acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criar pastas do dataset se não existirem
os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True) # Garante que a pasta do ChromaDB exista

# Carregar Modelo de ML
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado com sucesso.")
except FileNotFoundError:
    print(f"[API Startup] AVISO: Modelo '{MODEL_PATH}' não encontrado. O endpoint /analyze/ não funcionará.")
except Exception as e:
    print(f"[API Startup] ERRO ao carregar o modelo: {e}")

# Inicializar ChromaDB Client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
print(f"[API Startup] ChromaDB coleção '{CHROMA_COLLECTION_NAME}' conectada. Itens existentes: {chroma_collection.count()}")


# --- Funções de Extração de Features ---
def extract_features(audio_path: str) -> dict | None:
    """Extrai features acústicas de um arquivo de áudio."""
    try:
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)

        jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call(pulses, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        
        return {
            'jitter_local': jitter_local,
            'shimmer_local': shimmer_local,
            'mean_pitch': mean_pitch,
            'mean_hnr': mean_hnr
        }
    except Exception as e:
        print(f"[ERRO] Erro ao extrair features de {audio_path}: {e}")
        return None

# --- Função de Geração de Embedding para ChromaDB ---
def get_feature_embedding(features_dict: dict) -> list[float]:
    """Converte um dicionário de features em uma lista de floats para o ChromaDB."""
    ordered_features = [
        features_dict.get('jitter_local', 0.0),
        features_dict.get('shimmer_local', 0.0),
        features_dict.get('mean_pitch', 0.0),
        features_dict.get('mean_hnr', 0.0)
    ]
    return ordered_features

# --- Endpoints da API ---

@app.get("/", tags=["Health"])
async def read_root():
    """Endpoint de verificação de saúde da API."""
    return {"status": "ok", "message": "Bem-vindo à API da angel.ia"}

@app.post("/analyze/", response_model=AnalysisReport, tags=["Analysis"])
async def analyze_voice(audio_file: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio, extrai features, faz uma predição e retorna um relatório.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modelo de análise indisponível no servidor.")

    temp_audio_path = f"temp_{uuid.uuid4()}_{audio_file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        features = extract_features(temp_audio_path)
        if not features:
            raise HTTPException(status_code=400, detail="Não foi possível processar o arquivo de áudio.")

        features_df = pd.DataFrame([features])
        features_df = features_df[['jitter_local', 'shimmer_local', 'mean_pitch', 'mean_hnr']]
        
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = float(max(probabilities))

        risk_level = "Risco Detectado" if prediction == 1 else "Normal"
        
        # TODO: Integrar LLM e VectorDB para uma recomendação mais dinâmica aqui
        recommendation = "Recomenda-se acompanhamento com um profissional de saúde para uma avaliação completa." if risk_level == "Risco Detectado" else "Os biomarcadores vocais analisados estão dentro da normalidade."

        return AnalysisReport(
            riskLevel=risk_level,
            confidence=round(confidence, 2),
            recommendation=recommendation,
            biomarkers={k: f"{v:.4f}" for k, v in features.items()}
        )
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro no endpoint /analyze/: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/add-to-dataset/", response_model=AddToDatasetResponse, tags=["Dataset"])
async def add_to_dataset(
    diagnosis: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Recebe um áudio e metadados, extrai features e salva no dataset (áudio, CSV e ChromaDB).
    """
    file_extension = os.path.splitext(audio_file.filename)[1]
    if not file_extension: # Caso o arquivo não tenha extensão (ex: webm sem .webm)
        file_extension = ".webm" # Assume .webm como padrão para gravações web
    
    # Gera um nome de arquivo único para salvar o áudio
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    audio_save_path = os.path.join(DATASET_AUDIO_DIR, unique_filename)
    
    try:
        # 1. Salvar o arquivo de áudio de forma permanente
        with open(audio_save_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Dataset] Áudio salvo em: {audio_save_path}")

        # 2. Extrair as features do áudio salvo
        features = extract_features(audio_save_path)
        if not features:
            # Se a extração falhar, removemos o áudio para não ter dados inconsistentes
            os.remove(audio_save_path)
            raise HTTPException(status_code=400, detail="Falha ao extrair features do áudio. O arquivo não foi salvo.")
        print(f"[Dataset] Features extraídas: {features}")

        # 3. Preparar a nova linha de dados para o arquivo CSV (para inspeção humana)
        dataset_entry_id = str(uuid.uuid4()) # ID único para a entrada do dataset
        new_data_row = {
            'id': dataset_entry_id, # ID único para a entrada
            'audio_filename': unique_filename,
            'diagnosis': diagnosis,
            'age': age,
            'gender': gender,
            'timestamp': datetime.now().isoformat(),
            **features  # Adiciona todas as features extraídas
        }
        
        new_df = pd.DataFrame([new_data_row])

        # Adicionar (append) a nova linha ao arquivo CSV
        file_exists = os.path.isfile(DATASET_CSV_PATH)
        new_df.to_csv(DATASET_CSV_PATH, mode='a', header=not file_exists, index=False)
        print(f"[Dataset] Metadados e features adicionados ao CSV: {DATASET_CSV_PATH}")

        # 4. Adicionar ao ChromaDB (banco de dados vetorial)
        embedding = get_feature_embedding(features)
        chroma_metadata = {
            'id': dataset_entry_id,
            'audio_filename': unique_filename,
            'diagnosis': diagnosis,
            'age': age,
            'gender': gender,
            'timestamp': datetime.now().isoformat(),
            **features # Incluir as features nos metadados do Chroma para fácil consulta
        }
        
        chroma_collection.add(
            embeddings=[embedding],
            metadatas=[chroma_metadata],
            documents=[f"Audio recording for dataset: {unique_filename} - Diagnosis: {diagnosis}"],
            ids=[dataset_entry_id]
        )
        print(f"[Dataset] Dados adicionados ao ChromaDB com ID: {dataset_entry_id}. Total no DB: {chroma_collection.count()}")

        return AddToDatasetResponse(
            message="Dados adicionados ao dataset (áudio, CSV, ChromaDB) com sucesso!",
            audio_filename=unique_filename,
            dataset_entry_id=dataset_entry_id,
            features_count=len(features)
        )
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro no endpoint /add-to-dataset/: {e}")
        # Tenta remover o áudio salvo se houver erro após salvá-lo
        if os.path.exists(audio_save_path):
            os.remove(audio_save_path)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar o áudio: {e}")