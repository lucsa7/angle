FROM python:3.10-slim

# Instala git y dependencias de OpenCV
RUN apt-get update \
 && apt-get install -y git libgl1-mesa-glx libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Render inyecta el puerto en $PORT
ENV PORT ${PORT:-8050}
EXPOSE 8050

# Arranca con Gunicorn y expande $PORT en shell
CMD gunicorn app:server --bind 0.0.0.0:$PORT --log-file -




