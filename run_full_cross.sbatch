#!/bin/bash
#SBATCH --job-name=drawspeech_cross
#SBATCH --partition=a100         # or rtx3080 / a100 – adjust to your cluster
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=cross_rmse_%j.out

# ---------------------------------------------------------
# Cross-utterance sketch evaluation on the full LJSpeech test set
# ---------------------------------------------------------

set -e

echo "=== Load environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drawspeech

# Set proxy if needed (uncomment and adjust)
# export http_proxy=http://proxy.nhr.fau.de:80
# export https_proxy=http://proxy.nhr.fau.de:80

export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

# ---------------------------------------------------------
# 1. Generate cross-utterance sketches for all 300 test files
# ---------------------------------------------------------
echo "=== Generating cross-utterance sketches for full test set ==="
python - <<'PYEOF'
import os, json, random, numpy as np
from scipy.signal import savgol_filter

# Load test set metadata (from full preprocessing)
with open('data/dataset/metadata/ljspeech/phoneme_level/metadata.txt') as f:
    test_lines = [l.strip().split('|') for l in f if l.strip()]

# Get list of all basenames from full dataset (train+test) – we'll use training set as source pool
# The train/val/test split is defined in data/dataset/metadata/ljspeech/
# We'll load the train.json and val.json to create a source pool
source_pool = []
for split in ['train.json', 'val.json']:
    with open(f'data/dataset/metadata/ljspeech/{split}') as f:
        split_data = json.load(f)['data']
        for item in split_data:
            # Extract basename from wav path (e.g., "wavs/LJ001-0001.wav")
            basename = os.path.splitext(os.path.basename(item['wav']))[0]
            source_pool.append(basename)

random.seed(42)

cross_dir = 'data/dataset/metadata/ljspeech/cross_sketches'
os.makedirs(cross_dir, exist_ok=True)

for parts in test_lines:
    target = parts[0]
    # Choose a random source that is NOT the same as target
    possible = [b for b in source_pool if b != target]
    source = random.choice(possible)

    # Load source pitch (phoneme-level, full dataset)
    src_pitch_path = f'data/dataset/metadata/ljspeech/phoneme_level/pitch/LJSpeech-pitch-{source}.npy'
    if not os.path.exists(src_pitch_path):
        # fallback to any if missing
        continue
    src_pitch = np.load(src_pitch_path)

    # Smooth to create sketch
    window = min(5, len(src_pitch))
    if window % 2 == 0:
        window += 1
    if window >= 3:
        sketch_src = savgol_filter(src_pitch, window_length=window, polyorder=2)
    else:
        sketch_src = src_pitch

    # Target phoneme count
    target_phonemes = parts[2].strip('{}').split()
    target_n = len(target_phonemes)

    # Resample sketch to target length
    x_old = np.linspace(0, 1, len(sketch_src))
    x_new = np.linspace(0, 1, target_n)
    sketch_target = np.interp(x_new, x_old, sketch_src)

    np.save(f'{cross_dir}/{target}_cross_sketch.npy', sketch_target)

print(f"Cross-sketches saved for {len(test_lines)} targets.")
PYEOF

# ---------------------------------------------------------
# 2. Build inference JSON for the full test set with cross sketches
# ---------------------------------------------------------
echo "=== Building inference JSON ==="
python - <<'PYEOF'
import json, os

with open('data/dataset/metadata/ljspeech/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

data = []
for parts in lines:
    basename, speaker, phonemes, raw_text = parts
    data.append({
        'wav': f'data/dataset/LJSpeech-1.1/wavs/{basename}.wav',
        'transcription': raw_text,
        'phonemes': phonemes,
        'pitch_sketch': f'data/dataset/metadata/ljspeech/cross_sketches/{basename}_cross_sketch.npy',
        'energy_sketch': f'data/dataset/metadata/ljspeech/phoneme_level/energy/LJSpeech-energy-{basename}.npy',
        'pitch': '',
        'energy': '',
        'duration': None,
        'pitch_length': None,
        'energy_length': None
    })

with open('tests/inference_full_cross.json', 'w') as f:
    json.dump({'data': data}, f, indent=2)

print(f"Created tests/inference_full_cross.json with {len(data)} samples.")
PYEOF

# ---------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------
echo "=== Running inference on full test set (this will take a while) ==="
python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_full_cross.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

# ---------------------------------------------------------
# 4. Compute RMSE
# ---------------------------------------------------------
echo "=== Computing RMSE ==="
LATEST_DIR=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
python compute_rmse.py \
    --generated_dir "$LATEST_DIR" \
    --original_dir data/dataset/LJSpeech-1.1/wavs