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
# CHROMA_DB_PATH = str(BASE_DIR / "chroma_db") # Descomentar se for usar ChromaDB
# CHROMA_COLLECTION_NAME = "audio_features_collection" # Descomentar se for usar ChromaDB

# --- Segurança (para deploy, ajuste isso!) ---
# Pega a URL do frontend da variável de ambiente, ou usa localhost como fallback
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [
    FRONTEND_URL,          # URL do seu frontend na Render
    "http://localhost:3000", # Para desenvolvimento local do frontend
    "http://localhost:8000"  # Para testes locais da API
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
    version="1.4.0" # Versão atualizada
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Cria os diretórios necessários se não existirem
os.makedirs(DATASET_AUDIO_DIR, exist_ok=True)
# os.makedirs(CHROMA_DB_PATH, exist_ok=True) # Descomentar se for usar ChromaDB

# Carrega o modelo treinado
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo '{MODEL_PATH}' carregado com sucesso.")
except Exception as e:
    print(f"[API Startup] AVISO: Modelo '{MODEL_PATH}' não carregado. Erro: {e}. A funcionalidade de análise pode estar limitada.")

# --- Funções de Extração de Features (ATUALIZADA E ROBUSTA) ---
def extract_features(audio_path: str) -> dict | None:
    """
    Extrai features fonéticas de um arquivo de áudio usando Parselmouth.
    Retorna um dicionário de features ou None em caso de falha.
    """
    try:
        sound = parselmouth.Sound(audio_path)

        # Verificação mínima de duração do áudio para features complexas
        # Parselmouth precisa de um mínimo de som para detectar ciclos de voz
        MIN_DURATION_FOR_COMPLEX_FEATURES = 0.5 # segundos
        if sound.duration < MIN_DURATION_FOR_COMPLEX_FEATURES:
            print(f"[ERRO - Parselmouth] Áudio muito curto ({sound.duration:.2f}s) para extrair features complexas.")
            return None

        # Parâmetros de pitch (ajustáveis conforme o tipo de voz esperado)
        # Valores de 75 a 600 Hz são para vozes adultas, ajuste se necessário.
        f0min, f0max = 75, 600

        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)
        
        # Jitter e Shimmer dependem da detecção de ciclos de voz (PointProcess).
        # Esta etapa é a mais sensível a silêncio ou fala atípica.
        jitter_local = 0.0
        shimmer_local = 0.0
        try:
            pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            if pulses.get_number_of_points() > 0:
                jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer_local = parselmouth.praat.call(pulses, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            else:
                print("[AVISO - Parselmouth] 'To PointProcess' não detectou ciclos de voz periódicos. Jitter/Shimmer serão 0.")
        except Exception as e:
            print(f"[AVISO - Parselmouth] Falha ao calcular Jitter/Shimmer: {e}. Serão definidos como 0.")
            # Continuamos a execução, definindo os valores como 0.0

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
        # Comando ffmpeg para converter para WAV, 16-bit PCM, 1 canal (mono), 44.1kHz sample rate
        command = [
            "ffmpeg",
            "-i", str(input_path),      # Arquivo de entrada
            "-acodec", "pcm_s16le",     # Codec de áudio (WAV padrão)
            "-ac", "1",                 # 1 canal de áudio (mono)
            "-ar", "44100",             # Sample rate de 44.1kHz
            "-y",                       # Sobrescrever arquivo de saída se existir
            str(output_path)
        ]
        # Use capture_output=True para pegar stderr em caso de erro
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
        # 1. Salvar o arquivo de áudio recebido temporariamente
        temp_input_path = BASE_DIR / f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Análise] Áudio recebido salvo temporariamente em: {temp_input_path.name}")

        # 2. Converter para WAV se não for WAV (ou para garantir compatibilidade)
        final_audio_path = temp_input_path
        if audio_file.content_type != "audio/wav":
            temp_wav_path = BASE_DIR / f"temp_{uuid.uuid4()}.wav"
            if not convert_audio_to_wav(temp_input_path, temp_wav_path):
                raise HTTPException(status_code=400, detail="Falha ao converter áudio para o formato WAV necessário.")
            final_audio_path = temp_wav_path
        else:
            print("[Análise] Áudio já é WAV, pulando conversão.")

        # 3. Extrair as features
        print(f"[Análise] Extraindo features de '{final_audio_path.name}'...")
        features = extract_features(str(final_audio_path))
        if features is None:
            raise HTTPException(status_code=400, detail="Não foi possível extrair features do áudio fornecido. Garanta que o áudio contém fala clara e com duração suficiente.")
        print(f"[Análise] Features extraídas: {features}")

        # 4. Fazer a predição (ajuste conforme seu modelo e features)
        features_df = pd.DataFrame([features])
        
        # Garantir que as colunas do DataFrame coincidam com as usadas no treinamento do modelo
        # Se seu modelo foi treinado com um StandardScaler, aplique-o aqui
        # Para demonstração, vamos apenas prever.
        
        # Exemplo: Se o modelo espera 'jitter_local', 'shimmer_local', 'mean_pitch', 'mean_hnr'
        # e eles são as únicas features que você usa, o df está ok.
        
        prediction = model.predict(features_df)[0]
        # Supondo que o modelo retorne 0 para saudável, 1 para Parkinson
        risk_level = "Baixo" if prediction == 0 else "Alto"
        recommendation = "Nenhum risco detectado. Mantenha os exames de rotina." if prediction == 0 else "Risco detectado. Recomendamos procurar um especialista para avaliação."
        
        # Para obter a confiança, se o modelo suportar (e.g., SVC com probability=True, RandomForest)
        confidence_score = 0.5 # Valor padrão se não houver predict_proba
        try:
            proba = model.predict_proba(features_df)[0]
            confidence_score = max(proba) # Confiança na classe prevista
            # Ajuste para mostrar a confiança na classe de "risco" se houver
            if risk_level == "Alto":
                # Supondo que a classe 1 seja "alto risco"
                confidence_score = proba[1]
            else:
                # Supondo que a classe 0 seja "baixo risco"
                confidence_score = proba[0]
            
        except AttributeError:
            print("[Análise] Modelo não suporta predict_proba para cálculo de confiança.")
            pass # Continua com a confiança padrão ou apenas 0.5
        
        return AnalysisReport(
            riskLevel=risk_level,
            confidence=confidence_score,
            recommendation=recommendation,
            biomarkers=features # Retorna as features extraídas como biomarcadores
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERRO GERAL] Ocorreu um erro inesperado em /analyze/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e}")
    finally:
        # Limpeza: Remove arquivos temporários
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
    Processa o áudio, extrai features e salva.
    """
    temp_input_path = None
    temp_wav_path = None
    
    try:
        # 1. Salvar o arquivo de áudio recebido temporariamente
        temp_input_path = BASE_DIR / f"temp_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[Dataset] Áudio recebido salvo temporariamente em: {temp_input_path.name}")

        # 2. Converter o áudio para .wav para extração de features
        # Mesmo que já seja WAV, converter garante o formato PCM de 16-bit
        temp_wav_path = BASE_DIR / f"temp_{uuid.uuid4()}.wav"
        if not convert_audio_to_wav(temp_input_path, temp_wav_path):
            raise HTTPException(status_code=400, detail="Falha ao converter áudio para o formato WAV necessário.")
        
        # 3. Extrair as features do arquivo .wav convertido
        print(f"[Dataset] Extraindo features de '{temp_wav_path.name}'...")
        features = extract_features(str(temp_wav_path)) # Passa o caminho como string para parselmouth
        if features is None:
            raise HTTPException(status_code=400, detail="Falha ao extrair features do áudio convertido. Garanta que o áudio contém fala clara e com duração suficiente.")
        print(f"[Dataset] Features extraídas: {features}")

        # 4. Salvar o áudio original (e.g., .webm) de forma permanente no dataset
        file_extension = Path(audio_file.filename).suffix or '.webm'
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        permanent_audio_path = DATASET_AUDIO_DIR / unique_filename
        
        # Move o arquivo temporário de entrada para o local permanente
        shutil.move(temp_input_path, permanent_audio_path)
        temp_input_path = None # Marca como movido para não ser deletado no 'finally'
        print(f"[Dataset] Áudio original salvo permanentemente em: {permanent_audio_path.name}")
        
        # 5. Salvar metadados e features no CSV
        dataset_entry_id = str(uuid.uuid4())
        new_data_row = {
            'id': dataset_entry_id,
            'audio_filename': unique_filename,
            'diagnosis': diagnosis,
            'age': age,
            'gender': gender,
            'timestamp': datetime.now().isoformat(),
            **features
        }
        new_df = pd.DataFrame([new_data_row])
        file_exists = DATASET_CSV_PATH.is_file()
        new_df.to_csv(DATASET_CSV_PATH, mode='a', header=not file_exists, index=False)
        print(f"[Dataset] Dados e features salvos no CSV.")

        # (Lógica do ChromaDB omitida para simplificar e focar na funcionalidade principal)
        
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
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e}")
    finally:
        # 6. Limpeza: Garante que os arquivos temporários sejam sempre removidos
        if temp_input_path and temp_input_path.exists():
            os.remove(temp_input_path)
            print(f"[Limpeza] Arquivo temporário de entrada removido: {temp_input_path.name}")
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)
            print(f"[Limpeza] Arquivo temporário WAV removido: {temp_wav_path.name}")