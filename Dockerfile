FROM python:3.10-slim

# Instala librerías necesarias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    ffmpeg build-essential cmake unzip pkg-config && \
    apt-get clean

# Crea carpeta y copia archivos
WORKDIR /app
COPY . .

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone puerto 8080
EXPOSE 8080

# Ejecuta el servidor
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080"]

