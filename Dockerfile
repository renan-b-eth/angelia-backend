# Use uma imagem base Python oficial com Debian (mais completa e compatível)
FROM python:3.10-slim-buster # 'slim-buster' é Debian 10, leve mas completa

# 1. Instale as dependências do sistema operacional
# Para ffmpeg: adicionamos repositórios e instalamos
# Para postgresql-dev: instalamos o libpq-dev
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 4. Instale as dependências Python *antes* de copiar o resto do código
# Use --upgrade pip para garantir a versão mais recente e evitar problemas
# O parâmetro --default-timeout=1000 aumenta o timeout para downloads, se for o caso
RUN pip install --upgrade pip --default-timeout=1000 && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 5. Copie todo o resto do seu código da API para o container
COPY . .

# 6. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 7. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]