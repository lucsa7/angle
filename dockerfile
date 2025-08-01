FROM python:3.10-slim

# Instala git y dependencias para OpenCV
RUN apt-get update \
 && apt-get install -y git libgl1-mesa-glx libglib2.0-0 ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Define directorio de trabajo
WORKDIR /app

# Copia todos los archivos
COPY . /app

# Instalación de dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Puerto (Render lo inyecta como $PORT)
ENV PORT=10000
EXPOSE 10000

# Arranque con Gunicorn (formato shell OK)
CMD gunicorn app:server --bind 0.0.0.0:$PORT --log-file -