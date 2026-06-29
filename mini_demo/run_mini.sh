#!/bin/bash
set -e

MODE=$1
if [[ "$MODE" != "phoneme" && "$MODE" != "word" && "$MODE" != "cross" ]]; then
    echo "Usage: bash mini_demo/run_mini.sh [phoneme|word|cross]"
    exit 1
fi

export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

# ---------- Ensure mini features exist (first-run or incomplete) ----------
if [ ! -f mini_demo/features/phoneme_level/pitch/LJSpeech-pitch-LJ003-0040.npy ]; then
    echo "=== Mini preprocessing not found or incomplete – running it now ==="

    # 1. Extract pitch/energy/duration from the 10 recordings
    python drawspeech/utilities/preprocessor/preprocessor.py drawspeech/config/preprocess_mini.yaml

    # 2. Create metadata.txt (the preprocessor doesn't write this file automatically)
    python -c "
import os, glob
from g2p_en import G2p
pitch_dir = 'mini_demo/features/phoneme_level/pitch'
files = sorted(glob.glob(os.path.join(pitch_dir, 'LJSpeech-pitch-*.npy')))
lines = []
g2p = G2p()
for f in files:
    basename = os.path.basename(f).replace('LJSpeech-pitch-', '').replace('.npy', '')
    with open('mini_demo/LJSpeech-1.1/metadata.csv') as meta:
        for row in meta:
            parts = row.strip().split('|')
            if parts[0] == basename:
                transcription = parts[2] if len(parts)>2 else ''
                break
        else:
            transcription = ''
    phones = g2p(transcription)
    phoneme_str = '{' + ' '.join(phones) + '}'
    lines.append(f'{basename}|LJSpeech|{phoneme_str}|{transcription}')
with open('mini_demo/features/phoneme_level/metadata.txt', 'w') as f:
    for line in lines:
        f.write(line + '\n')
print('metadata.txt created')
"

    # 3. Build word‑level sketches
    python mini_demo/scripts/create_word_sketches.py

    # 4. Build cross‑utterance sketches (for the 'cross' mode)
    python mini_demo/scripts/create_cross_sketches_all.py

    # 5. Generate the inference JSONs (phoneme, word, cross)
    python mini_demo/scripts/build_inference_jsons.py
    python mini_demo/scripts/build_cross_json_all.py

    echo "=== Mini preprocessing complete ==="
fi

# ---------- Ensure stats.json is accessible ----------
mkdir -p data/dataset/metadata/ljspeech/phoneme_level
cp mini_demo/features/phoneme_level/stats.json data/dataset/metadata/ljspeech/phoneme_level/stats.json 2>/dev/null || true

# ---------- Check for fixed checkpoint ----------
if [ ! -f data/checkpoints/drawspeech_fixed.ckpt ]; then
    echo "Fixed checkpoint not found. Run 'bash download_checkpoints.sh' first."
    exit 1
fi

case $MODE in
    phoneme)
        JSON=mini_demo/inference_phoneme.json
        ;;
    word)
        JSON=mini_demo/inference_word.json
        ;;
    cross)
        JSON=mini_demo/inference_cross_all.json
        ;;
esac

echo "=== Running $MODE-level inference ==="
python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference "$JSON" \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

LATEST_DIR=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)

echo ""
echo "=== RMSE for $MODE ==="
python mini_demo/scripts/compute_rmse.py \
    --generated_dir "$LATEST_DIR" \
    --original_dir mini_demo/LJSpeech-1.1/wavs