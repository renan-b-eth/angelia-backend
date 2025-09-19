# angelia-backend/main.py (CÓDIGO ATUALIZADO)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import boto3
from dotenv import load_dotenv
import uuid
import librosa
import numpy as np
import pandas as pd
from scipy.stats import variation # Para coeficiente de variação de pitch
import joblib # Para carregar o modelo

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir qualquer origem para desenvolvimento. Altere para o seu frontend em produção!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações do Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None

# Inicializar cliente S3 (R2)
s3_client = None
if all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL]):
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'
        )
        print("Conectado ao Cloudflare R2.")
    except Exception as e:
        print(f"Erro ao conectar ao R2: {e}")
        s3_client = None
else:
    print("Credenciais do R2 incompletas. O upload para o R2 estará desativado.")


# --- Carregar o Modelo de ML e o Scaler ---
MODEL_PATH = "models/modelo_diagnostico_voz.pkl"
SCALER_PATH = "models/scaler_diagnostico_voz.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
DIAGNOSIS_MAPPING_PATH = "models/diagnosis_mapping.pkl"

model = None
scaler = None
feature_names = None
diagnosis_mapping = None
reverse_diagnosis_mapping = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    diagnosis_mapping = joblib.load(DIAGNOSIS_MAPPING_PATH)
    # Criar mapeamento reverso para converter o número predito de volta para o nome
    reverse_diagnosis_mapping = {v: k for k, v in diagnosis_mapping.items()}
    print("Modelo, Scaler, Feature Names e Mapeamento de Diagnósticos carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: Um ou mais arquivos .pkl não foram encontrados na pasta 'models/'. Certifique-se de que estão lá.")
except Exception as e:
    print(f"ERRO ao carregar o modelo ou scaler: {e}")

# --- Função de Extração de Features (DEVE SER IDÊNTICA AO COLAB) ---
def extract_advanced_features_for_prediction(audio_path: str, sr: int) -> dict | None:
    """
    Extrai features de áudio avançadas usando apenas Librosa e Scipy.
    IDÊNTICA à função usada no treinamento no Colab.
    """
    try:
        y, sr_lib = librosa.load(audio_path, sr=sr)

        if len(y) < sr_lib * 0.3:
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr_lib, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_lib)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)

        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr_lib)
        f0_voiced = f0[voiced_flag]

        pitch_mean, pitch_std, pitch_cv = 0, 0, 0
        if len(f0_voiced) > 0:
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
            pitch_cv = variation(f0_voiced)

        jitter_approx = pitch_cv 
        shimmer_approx = variation(rmse[0]) if len(rmse[0]) > 1 else 0
        hnr_proxy = 0.0 # Placeholder, conforme treinamento


        features = {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_cv': float(pitch_cv),
            'rmse_cv': float(shimmer_approx),
            'hnr_proxy': float(hnr_proxy),
            'spectral_centroid_mean': float(spectral_centroid_mean),
            'spectral_centroid_std': float(spectral_centroid_std),
            'zcr_mean': float(zcr_mean),
            'zcr_std': float(zcr_std),
            'rmse_mean': float(rmse_mean),
            'rmse_std': float(rmse_std),
        }

        for i in range(len(mfcc_mean)):
            features[f'mfcc_mean_{i}'] = float(mfcc_mean[i])
            features[f'mfcc_std_{i}'] = float(mfcc_std[i])

        return features

    except Exception as e:
        print(f"ERRO na extração de features para predição: {e}")
        return None


@app.get("/")
async def root():
    return {"message": "Bem-vindo à API Angelia Voice AI!"}

@app.post("/add-to-dataset/")
async def add_to_dataset(
    audio_file: UploadFile = File(...),
    patient_id: str = Form(...),
    diagnosis: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    task_type: str = Form(...),
    recording_environment: str = Form(...),
    symptoms: str = Form(...),
    medications: str = Form(...)
):
    if s3_client is None:
        raise HTTPException(status_code=500, detail="Serviço de armazenamento (R2) não configurado.")

    try:
        # Ler o conteúdo do arquivo
        audio_data = await audio_file.read()

        # Gerar um nome de arquivo único
        unique_id = uuid.uuid4().hex[:8]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

        # Sanitizar metadados para o nome do arquivo
        sanitized_patient_id = re.sub(r'[^\w.-]', '', patient_id)
        sanitized_diagnosis = re.sub(r'[^\w.-]', '', diagnosis)
        sanitized_age = re.sub(r'[^\w.-]', '', age)
        sanitized_gender = re.sub(r'[^\w.-]', '', gender)
        sanitized_task_type = re.sub(r'[^\w.-]', '', task_type)
        sanitized_symptoms = re.sub(r'[^\w.-]', '', symptoms).replace(' ', '_')
        sanitized_medications = re.sub(r'[^\w.-]', '', medications).replace(' ', '_')

        file_name = f"{sanitized_diagnosis}/{sanitized_patient_id}_{sanitized_age}_{sanitized_gender}_{sanitized_symptoms}_{sanitized_medications}_{sanitized_task_type}_{timestamp}_{unique_id}.wav"

        # Upar para o R2
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=file_name,
            Body=audio_data,
            ContentType="audio/webm" # Assumindo que o frontend envia webm
        )

        return {"message": f"Áudio {file_name} adicionado ao dataset com sucesso!", "file_url": f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}/{file_name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar áudio ao dataset: {e}")


# --- NOVO ENDPOINT: Diagnóstico por Voz ---
@app.post("/diagnose-voice/")
async def diagnose_voice(audio_file: UploadFile = File(...)):
    if model is None or scaler is None or feature_names is None or diagnosis_mapping is None:
        raise HTTPException(status_code=500, detail="O modelo de diagnóstico não está carregado. Verifique os logs do servidor.")

    try:
        # Salvar o áudio temporariamente para Librosa
        temp_audio_path = f"/tmp/uploaded_audio_{uuid.uuid4().hex}.webm" # Nome temporário
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Extrair features do áudio
        SAMPLE_RATE = 44100 # Deve ser a mesma do treinamento
        features_dict = extract_advanced_features_for_prediction(temp_audio_path, SAMPLE_RATE)

        # Limpar arquivo temporário
        os.remove(temp_audio_path)

        if features_dict is None:
            raise HTTPException(status_code=400, detail="Não foi possível extrair features do áudio. Verifique a qualidade ou duração do áudio.")

        # Converter features para DataFrame (na ordem correta)
        # É CRÍTICO que as features estejam na mesma ordem que o modelo foi treinado
        features_df = pd.DataFrame([features_dict])

        # Garantir que o DataFrame tem as mesmas colunas e na mesma ordem que 'feature_names'
        # Isso é fundamental para a predição
        missing_cols = set(feature_names) - set(features_df.columns)
        for c in missing_cols:
            features_df[c] = 0 # Adiciona colunas ausentes com valor 0 ou média

        # Ordenar as colunas para que correspondam à ordem de treinamento
        features_for_prediction = features_df[feature_names]

        # Padronizar as features usando o scaler treinado
        scaled_features = scaler.transform(features_for_prediction)

        # Fazer a predição
        prediction_encoded = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]

        # Converter o resultado numérico de volta para o nome do diagnóstico
        predicted_diagnosis = reverse_diagnosis_mapping.get(prediction_encoded, "Diagnóstico Desconhecido")

        # Obter probabilidades para todas as classes
        probabilities = {}
        for class_encoded, prob in enumerate(prediction_proba):
            diagnosis_name = reverse_diagnosis_mapping.get(class_encoded, f"Classe {class_encoded}")
            probabilities[diagnosis_name] = f"{prob:.4f}"

        return {
            "predicted_diagnosis": predicted_diagnosis,
            "probabilities": probabilities,
            "message": "Diagnóstico de voz realizado com sucesso."
        }

    except Exception as e:
        print(f"Erro no endpoint /diagnose-voice/: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar o áudio para diagnóstico: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)