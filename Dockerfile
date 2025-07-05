FROM python:3.10-slim

# Instala librerías necesarias del sistema
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    ffmpeg build-essential cmake unzip pkg-config && \
    apt-get clean

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080"]


