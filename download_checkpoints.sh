#!/bin/bash
# Download all required checkpoints and fix key names

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
wget -q https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json -O data/checkpoints/LJ_V1/config.json
wget -q https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_LJSpeech.pth.tar -O data/checkpoints/LJ_V1/generator_v1

echo "=== Fixing checkpoint key names ==="
python fix_checkpoint.py

echo "=== Done! ==="
