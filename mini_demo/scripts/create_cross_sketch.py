import numpy as np
from scipy.signal import savgol_filter

# source and target filenames (without extension)
source = "LJ013-0220"
target = "LJ003-0040"

# load source pitch (phoneme-level)
src_pitch = np.load(f"mini_demo/features/phoneme_level/pitch/LJSpeech-pitch-{source}.npy")

# smooth to get a sketch (paper uses Savitzky-Golay, window ~5, order 2)
# we need enough points for the filter, use min window length
window = min(5, len(src_pitch))
if window % 2 == 0:
    window += 1
if window < 3:
    sketch_src = src_pitch  # too short to smooth
else:
    sketch_src = savgol_filter(src_pitch, window_length=window, polyorder=2)

# load target phoneme count
with open("mini_demo/features/phoneme_level/metadata.txt") as f:
    for line in f:
        parts = line.strip().split('|')
        if parts[0] == target:
            target_phoneme_str = parts[2]
            break
target_n = len(target_phoneme_str.strip('{}').split())

# resample sketch to target length
src_n = len(sketch_src)
x_old = np.linspace(0, 1, src_n)
x_new = np.linspace(0, 1, target_n)
sketch_target = np.interp(x_new, x_old, sketch_src)

# save
out_dir = "mini_demo/features/cross_sketches"
import os
os.makedirs(out_dir, exist_ok=True)
np.save(f"{out_dir}/{target}_from_{source}_sketch.npy", sketch_target)
print(f"Saved cross sketch: {target}_from_{source}_sketch.npy")
print(f"Sketch shape: {sketch_target.shape}")
