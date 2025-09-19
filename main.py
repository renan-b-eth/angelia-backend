# angelia-backend/main.py (Versão Refatorada - Foco no R2 e Nomes Estruturados, SEM DB)
import os
import shutil
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import re
import boto3

load_dotenv()

# --- Constantes de Configuração ---
BASE_DIR = Path(__file__).parent

# --- Variáveis de Ambiente para Cloudflare R2 ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL_BASE = os.getenv("R2_PUBLIC_URL_BASE")

# Validação das variáveis de ambiente R2
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

# --- Segurança ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000"]

# --- Estruturas de Resposta ---
class AddToDatasetResponse(BaseModel):
    message: str
    saved_url: str # Agora retorna a URL do R2

# --- Inicialização da API ---
app = FastAPI(
    title="angel.ia Backend API - Coleta de Áudio R2",
    description="API para coleta de dados de áudio estruturados e salvamento no Cloudflare R2.",
    version="1.0.0-r2-only"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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

def sanitize_filename(name: str) -> str:
    """Limpa uma string para ser usada como parte de um nome de arquivo."""
    name = name.lower()
    name = re.sub(r'\s+', '_', name) # Substitui espaços por underscores
    name = re.sub(r'[^a-z0-9_.-]', '', name) # Permite letras, números, _, . e -
    return name

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API angel.ia para Coleta de Áudio (R2)."}

@app.post("/add-to-dataset/", response_model=AddToDatasetResponse)
async def add_to_dataset(
    # Campos base do formulário
    patient_id: str = Form(...), # Novo campo para um ID único do paciente
    diagnosis: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    
    # Novos campos baseados nos artigos científicos (recomendações)
    task_type: str = Form(...), # Ex: "vogal_a_sustentada", "leitura_frase", "s_fricativo"
    recording_environment: str = Form("desconhecido"), # Ex: "silencioso", "moderado", "barulhento"
    symptoms: str = Form("nenhum"), # Ex: "fadiga_leve,estresse_alto", "nenhum"
    medications: str = Form("nenhum"), # Ex: "levodopa", "nenhum"
    
    audio_file: UploadFile = File(...)
):
    temp_input_path, temp_wav_path = None, None
    try:
        print("[DEBUG] Iniciando processamento do áudio.")
        
        # 1. Salva o arquivo original (webm) temporariamente no container
        temp_input_path = BASE_DIR / f"temp_input_{uuid.uuid4()}_{audio_file.filename}"
        with temp_input_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"[DEBUG] WEBM temporário salvo em: {temp_input_path.name}")

        # 2. Converte o arquivo temporário (webm) para WAV usando ffmpeg no container
        temp_wav_path = BASE_DIR / f"temp_converted_{uuid.uuid4()}.wav"
        if not convert_to_wav_with_ffmpeg(temp_input_path, temp_wav_path):
            raise HTTPException(400, "Falha ao converter áudio para WAV.")
        print(f"[DEBUG] WAV convertido salvo em: {temp_wav_path.name}")
        
        # --- Construir o nome de arquivo e caminho no R2 ---
        sanitized_patient_id = sanitize_filename(patient_id)
        sanitized_task_type = sanitize_filename(task_type)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8] # ID curto para evitar colisões
        
        # Nome do arquivo no R2: {diagnostico}/{tipo_tarefa}/{id_paciente}_{idade}_{genero}_{sintomas}_{medicamentos}_{timestamp}_{uniqueid}.wav
        # A pasta principal é o diagnóstico para fácil separação
        r2_folder_path = sanitize_filename(diagnosis) # Ex: "parkinson", "saudavel"
        
        # Exemplo de nome de arquivo: "parkinson_paciente123_35_masculino_sintomas_nenhum_meds_nenhum_vogal_a_20240101123000_abcde123.wav"
        filename_parts = [
            sanitized_patient_id,
            str(age),
            gender,
            sanitize_filename(symptoms),
            sanitize_filename(medications),
            sanitized_task_type,
            timestamp_str,
            unique_id
        ]
        
        r2_object_key = f"{r2_folder_path}/{'_'.join(filename_parts)}.wav"
        
        print(f"[DEBUG] Tentando upload para R2. Bucket: {R2_BUCKET_NAME}, Key: {r2_object_key}")
        
        s3_client.upload_file(str(temp_wav_path), R2_BUCKET_NAME, r2_object_key)
        
        print(f"[DEBUG] Upload para R2 bem-sucedido para key: {r2_object_key}")
        
        audio_r2_url = f"{R2_PUBLIC_URL_BASE}/{r2_object_key}"
        print(f"[DEBUG] URL pública do R2: {audio_r2_url}")

        return AddToDatasetResponse(
            message="Áudio adicionado ao dataset no Cloudflare R2 com sucesso!",
            saved_url=audio_r2_url
        )
    except Exception as e:
        print(f"[ERRO R2/Interno] Falha no processamento: {e}")
        raise HTTPException(500, f"Erro interno no servidor ao processar o áudio: {e}")
    finally:
        print("[DEBUG] Iniciando limpeza de arquivos temporários.")
        if temp_input_path and temp_input_path.exists():
            os.remove(temp_input_path)
            print(f"[DEBUG] Removido temp_input_path: {temp_input_path.name}")
        if temp_wav_path and temp_wav_path.exists():
            os.remove(temp_wav_path)
            print(f"[DEBUG] Removido temp_wav_path: {temp_wav_path.name}")
        print("[DEBUG] Limpeza de arquivos temporários concluída.")