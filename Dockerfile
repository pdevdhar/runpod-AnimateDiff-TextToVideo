FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install essential video/image system dependencies missing from base images
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx git && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create the symlink to redirect local model searches to the persistent volume
RUN ln -s /runpod-volume/models /models

CMD [ "python", "-u", "/server.py" ]
