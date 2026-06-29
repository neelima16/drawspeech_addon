#!/bin/bash
set -e

echo "=== Downloading DrawSpeech checkpoints ==="
mkdir -p data/checkpoints/LJ_V1

pip install huggingface_hub -q
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('HappyColor/DrawSpeech', 'vae.ckpt', local_dir='data/checkpoints')
hf_hub_download('HappyColor/DrawSpeech', 'drawspeech.ckpt', local_dir='data/checkpoints')
"

echo "=== Downloading HiFi-GAN vocoder ==="
curl -L -o data/checkpoints/LJ_V1/config.json https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json
curl -L -o data/checkpoints/LJ_V1/generator_v1 "https://drive.usercontent.google.com/download?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download"

echo "=== Cloning and installing taming-transformers ==="
if [ ! -d "taming-transformers" ]; then
    git clone https://github.com/CompVis/taming-transformers.git
    cd taming-transformers && pip install -e . && cd ..
else
    echo "taming-transformers already exists, skipping clone."
fi

echo "=== Fixing checkpoint key names ==="
python fix_checkpoint.py

echo "=== Done! ==="
