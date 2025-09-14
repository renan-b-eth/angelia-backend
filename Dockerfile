# angelia-backend/Dockerfile - Versão Final para isolar parselmouth e resolver googleads

# Use uma imagem base Python oficial com Debian Bullseye (mais recente e mantida)
FROM python:3.10-slim-bullseye

# Definição de variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# 1. Instalar as dependências do sistema operacional
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

# 4. ATUALIZAÇÃO CRÍTICA: Atualizar as ferramentas de build ANTES de instalar os pacotes
RUN pip install --upgrade pip setuptools wheel

# 5. Instale parselmouth SEPARADAMENTE e sem suas dependências (muito arriscado, mas necessário para teste)
#    Se parselmouth sozinho causar o erro googleads, ele é o problema.
#    Esta etapa pode falhar se parselmouth tiver dependências compiladas que precisamos.
RUN pip install --no-cache-dir --only-binary :all: parselmouth==1.1.1 || pip install --no-cache-dir parselmouth==1.1.1

# 6. Remova parselmouth do requirements.txt temporariamente para o próximo passo.
#    Isso é para evitar que pip tente instalar de novo, mas a versão que falhou será ignorada.
RUN sed -i '/parselmouth/d' requirements.txt

# 7. Instale o RESTO das dependências Python a partir do requirements.txt
#    A flag --only-binary :all: foi removida, se não foi aplicada no parselmouth.
RUN pip install -r requirements.txt

# 8. Copie todo o resto do seu código da API para o container
COPY . .

# 9. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 10. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]