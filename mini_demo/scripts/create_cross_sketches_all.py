import numpy as np, os, random
from scipy.signal import savgol_filter

with open('mini_demo/features/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

# list of basenames
basenames = [l[0] for l in lines]

# For each target, pick a different source randomly (but not itself)
random.seed(42)
cross_dir = 'mini_demo/features/cross_sketches'
os.makedirs(cross_dir, exist_ok=True)

for parts in lines:
    target = parts[0]
    # choose a source that is not the target
    possible_sources = [b for b in basenames if b != target]
    source = random.choice(possible_sources)

    # load source pitch
    src_pitch = np.load(f'mini_demo/features/phoneme_level/pitch/LJSpeech-pitch-{source}.npy')
    window = min(5, len(src_pitch))
    if window % 2 == 0: window += 1
    if window >= 3:
        sketch_src = savgol_filter(src_pitch, window_length=window, polyorder=2)
    else:
        sketch_src = src_pitch

    # target phoneme count
    target_phonemes = parts[2].strip('{}').split()
    target_n = len(target_phonemes)

    # resample sketch to target length
    x_old = np.linspace(0, 1, len(sketch_src))
    x_new = np.linspace(0, 1, target_n)
    sketch_target = np.interp(x_new, x_old, sketch_src)

    np.save(f'{cross_dir}/{target}_cross_sketch.npy', sketch_target)

print("Cross-sketches generated for all 10 targets.")
