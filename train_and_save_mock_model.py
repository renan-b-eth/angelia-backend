# angelia-backend/train_and_save_mock_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Gerando dados de treinamento falsos...")
data = {
    'jitter_local': [0.005, 0.009, 0.004, 0.012, 0.006, 0.015, 0.003],
    'shimmer_local': [0.03, 0.08, 0.02, 0.1, 0.04, 0.12, 0.01],
    'mean_pitch': [120.5, 150.2, 110.8, 180.1, 135.9, 190.0, 105.0],
    'mean_hnr': [20.1, 15.3, 22.5, 12.8, 18.9, 11.0, 24.0],
    'diagnosis': [0, 1, 0, 1, 0, 1, 0] # 0: Normal, 1: Risco
}
df = pd.DataFrame(data)

X = df[['jitter_local', 'shimmer_local', 'mean_pitch', 'mean_hnr']]
y = df['diagnosis']

print("Treinando um modelo RandomForestClassifier simples...")
model = RandomForestClassifier(n_estimators=5, random_state=42)
model.fit(X, y)

print(f"Acurácia do modelo de teste: {model.score(X, y):.2f}")

if not os.path.exists('models'):
    os.makedirs('models')

model_path = 'models/modelo_svm.pkl' # Usamos .pkl para compatibilidade com a API
joblib.dump(model, model_path)

print(f"Modelo de teste salvo em: '{model_path}'")