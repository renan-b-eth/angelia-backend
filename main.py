import librosa
import numpy as np
import io
import os # Adicione esta importação
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
import joblib
# ... suas outras importações

# Crie uma instância do FastAPI
app = FastAPI()

# Defina a pasta para o dataset
DATASET_DIR = "dataset"
AUDIO_DIR = os.path.join(DATASET_DIR, "audios")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")

# Cria as pastas se não existirem
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Carregamento do modelo (se existir)
try:
    MODEL_PATH = "modelo_svm.pkl" # Certifique-se de que este caminho está correto
    model = joblib.load(MODEL_PATH)
    print(f"[API Startup] Modelo carregado com sucesso de: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[API Startup] ATENÇÃO: Modelo '{MODEL_PATH}' não encontrado. A funcionalidade de análise pode estar limitada.")
    model = None
except Exception as e:
    print(f"[API Startup] ERRO ao carregar o modelo: {e}")
    model = None

# Função para extrair MFCCs (ou outras features)
def extract_features(audio_path_or_bytes, sr=None):
    print(f"[Feature Extraction] Iniciando extração de features...")
    y, sr = librosa.load(audio_path_or_bytes, sr=sr) # Tentar carregar
    print(f"[Feature Extraction] librosa.load - sr: {sr}, duração: {len(y)/sr:.2f}s")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print(f"[Feature Extraction] MFCCs extraídos. Shape: {mfccs.shape}")
    return np.mean(mfccs.T, axis=0) # Retorna a média das MFCCs

# Modelo Pydantic para adicionar ao dataset
class AddToDatasetRequest(BaseModel):
    diagnosis: str
    age: int
    gender: str

# Rota para adicionar áudio e metadados ao dataset
@app.post("/add-to-dataset/")
async def add_to_dataset(
    audio_file: UploadFile = File(...),
    diagnosis: str = "unknown", # Valor padrão para teste
    age: int = 0, # Valor padrão para teste
    gender: str = "other" # Valor padrão para teste
):
    print(f"[Add to Dataset] Recebida requisição para adicionar ao dataset.")
    print(f"[Add to Dataset] Arquivo: {audio_file.filename}, Tipo: {audio_file.content_type}")
    print(f"[Add to Dataset] Diagnóstico: {diagnosis}, Idade: {age}, Gênero: {gender}")

    # Gerar um nome de arquivo único
    file_extension = audio_file.filename.split(".")[-1] if "." in audio_file.filename else "webm"
    audio_filename = f"audio_{diagnosis}_{age}_{gender}_{len(os.listdir(AUDIO_DIR)) + 1}.{file_extension}"
    audio_full_path = os.path.join(AUDIO_DIR, audio_filename)

    try:
        # Salva o arquivo temporariamente para processamento
        print(f"[Add to Dataset] Tentando ler conteúdo do áudio...")
        contents = await audio_file.read()
        print(f"[Add to Dataset] Conteúdo lido. Tamanho: {len(contents)} bytes.")

        # ATENÇÃO: Pode ser necessário salvar o arquivo em disco para librosa.load em alguns casos.
        # Vamos tentar com BytesIO primeiro, mas se falhar, essa é a alternativa.
        # with open(audio_full_path, "wb") as f:
        #     f.write(contents)
        # print(f"[Add to Dataset] Áudio salvo temporariamente em: {audio_full_path}")
        
        # Extrair features
        print(f"[Add to Dataset] Chamando extract_features...")
        features = extract_features(io.BytesIO(contents), sr=None) # Ou sr=22050
        print(f"[Add to Dataset] Features extraídas com sucesso. Shape: {features.shape}")

        # Salvar features e metadados (simulando ou enviando para ChromaDB)
        # Exemplo simplificado:
        feature_filename = f"{os.path.splitext(audio_filename)[0]}.npy"
        np.save(os.path.join(FEATURES_DIR, feature_filename), features)
        
        # Aqui você integraria com ChromaDB
        print(f"[Dataset] Áudio salvo em: {audio_full_path}")
        print(f"[Dataset] Features salvas em: {os.path.join(FEATURES_DIR, feature_filename)}")
        print(f"[Dataset] Dados adicionados ao ChromaDB (simulado).") # Substituir por lógica real do ChromaDB

        return {"message": "Áudio e dados adicionados ao dataset com sucesso!", "audio_path": audio_full_path}

    except Exception as e:
        print(f"[Add to Dataset] ERRO FATAL ao processar áudio: {e}")
        # Se houve erro, garante que o arquivo temporário não fique por aí, se salvou
        # if os.path.exists(audio_full_path):
        #     os.remove(audio_full_path)
        raise HTTPException(status_code=400, detail=f"Ocorreu um erro interno ao processar o áudio: Falha ao extrair features do áudio. O arquivo não foi salvo. Detalhes: {e}")

# Rota de análise (exemplo similar)
@app.post("/analyze/")
async def analyze_audio(audio_file: UploadFile = File(...)):
    print(f"[Analyze] Recebida requisição para análise.")
    print(f"[Analyze] Arquivo: {audio_file.filename}, Tipo: {audio_file.content_type}")

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de análise não carregado ou não disponível.")

    try:
        print(f"[Analyze] Tentando ler conteúdo do áudio...")
        contents = await audio_file.read()
        print(f"[Analyze] Conteúdo lido. Tamanho: {len(contents)} bytes.")

        # Extrair features
        print(f"[Analyze] Chamando extract_features para análise...")
        features = extract_features(io.BytesIO(contents), sr=None) # Ou sr=22050
        print(f"[Analyze] Features extraídas para análise. Shape: {features.shape}")

        # Fazer a predição
        features_reshaped = features.reshape(1, -1) # Redimensiona para o modelo
        prediction = model.predict(features_reshaped)[0]
        prediction_proba = model.predict_proba(features_reshaped)[0] # Se o modelo tiver predict_proba

        print(f"[Analyze] Predição: {prediction}")
        return {"prediction": prediction, "confidence": prediction_proba.tolist()}

    except Exception as e:
        print(f"[Analyze] ERRO FATAL ao analisar áudio: {e}")
        raise HTTPException(status_code=400, detail=f"Falha ao analisar áudio. Detalhes: {e}")

# Rota de teste
@app.get("/")
async def root():
    return {"message": "API Angelia está funcionando!"}