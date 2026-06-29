#!/bin/bash
set -e

MODE=$1
if [[ "$MODE" != "phoneme" && "$MODE" != "word" && "$MODE" != "cross" ]]; then
    echo "Usage: bash mini_demo/run_mini.sh [phoneme|word|cross]"
    exit 1
fi

export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

# Ensure the mini-demo stats are accessible at the path expected by the config
mkdir -p data/dataset/metadata/ljspeech/phoneme_level
cp mini_demo/features/phoneme_level/stats.json data/dataset/metadata/ljspeech/phoneme_level/stats.json 2>/dev/null || true

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
