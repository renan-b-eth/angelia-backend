# angelia-backend/Dockerfi# angelia-backend/Dockerf# angelia-backend/Dockerfile

# Use uma imagem base Python oficial com Debian Bullseye (mais recente e mantida)
# 'slim-bullseye' é Debian 11, leve e com repositórios ativos
FROM python:3.10-slim-bullseye

# 1. Instalar as dependências do sistema operacional
# Repositórios padrão do Bullseye já devem incluir tudo necessário.
# ffmpeg é o pacote binário
# libpq-dev é para PostgreSQL (psycopg2-binary)
# build-essential é para compilação (inclui gcc, g++, make)
# pkg-config e libsndfile1 são ferramentas de build/libs de áudio
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        pkg-config \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 4. Instale as dependências Python *antes* de copiar o resto do código
RUN pip install --upgrade pip --default-timeout=1000 && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 5. Copie todo o resto do seu código da API para o container
COPY . .

# 6. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 7. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]