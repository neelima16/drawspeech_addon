import os, json, numpy as np

# Load metadata to get word counts per sentence
with open('mini_demo/features/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

human_dir = 'mini_demo/features/human_sketches'
os.makedirs(human_dir, exist_ok=True)

# For each sentence, create a word-level pitch sketch with a chosen pattern
for idx, parts in enumerate(lines):
    basename, speaker, phonemes, raw_text = parts
    # Count words by splitting transcription (or using sp count from phonemes)
    words = raw_text.split()
    n_words = len(words)

    # Choose a pattern based on index (or fixed)
    if idx % 4 == 0:
        # Rising
        sketch_vals = np.linspace(0.2, 0.9, n_words)
    elif idx % 4 == 1:
        # Falling
        sketch_vals = np.linspace(0.9, 0.2, n_words)
    elif idx % 4 == 2:
        # Peak on third word (if exists)
        sketch_vals = np.full(n_words, 0.5)
        peak_idx = min(2, n_words-1)
        sketch_vals[peak_idx] = 0.9
    else:
        # Flat
        sketch_vals = np.full(n_words, 0.5)

    # Save as numpy array
    np.save(os.path.join(human_dir, f'{basename}_word_sketch.npy'), sketch_vals)

    # Also save a JSON entry for later use
    print(f'{basename}: {sketch_vals.tolist()} ({words})')

print(f'Created {len(lines)} synthetic word sketches in {human_dir}')
