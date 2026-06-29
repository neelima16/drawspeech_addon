import os
import numpy as np

with open('mini_demo/features/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

word_dir = 'mini_demo/features/word_level'
os.makedirs(word_dir, exist_ok=True)

for parts in lines:
    basename, speaker, phonemes_str, raw_text = parts
    pitch = np.load(f'mini_demo/features/phoneme_level/pitch/LJSpeech-pitch-{basename}.npy')
    energy = np.load(f'mini_demo/features/phoneme_level/energy/LJSpeech-energy-{basename}.npy')

    phonemes = phonemes_str.strip('{}').split()
    boundaries = [i for i, ph in enumerate(phonemes) if ph == 'sp']
    word_groups = []
    start = 0
    for end in boundaries:
        if end > start:
            word_groups.append((start, end))
        start = end + 1
    if start < len(pitch):
        word_groups.append((start, len(pitch)))

    word_pitch = np.array([pitch[s:e].mean() for (s, e) in word_groups])
    word_energy = np.array([energy[s:e].mean() for (s, e) in word_groups])

    np.save(os.path.join(word_dir, f'{basename}_word_pitch.npy'), word_pitch)
    np.save(os.path.join(word_dir, f'{basename}_word_energy.npy'), word_energy)

print("Word-level sketches saved to mini_demo/features/word_level/")
