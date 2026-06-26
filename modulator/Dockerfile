# Minimal CUDA image for Stage A extraction on RunPod.
# Deliberately NO robustbench/autoattack (the old dependency-conflict source).
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/modulator
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# datasets + caches live on the mounted network volume, not in the image
# (raw data is NOT baked; mount it and pass --volume)
ENV PYTHONUNBUFFERED=1
CMD ["bash"]
