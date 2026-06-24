import os
import numpy as np
import librosa
import pyworld as pw
from tqdm import tqdm

generated_dir = "log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_06-24-05:24_cfg_scale_3.5_ddim_200_n_cand_1"
original_wav_dir = "data/dataset/LJSpeech-1.1/wavs"
hop_length = 256
sr = 22050

def extract_pitch_and_energy(wav_path):
    """Extract frame-level pitch (Hz) and energy (dB)"""
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav.astype(np.float64)

    # Pitch
    f0, t = pw.dio(wav, sr, frame_period=hop_length/sr*1000)
    f0 = pw.stonemask(wav, f0, t, sr)

    # Energy: RMS using the same number of frames as pitch
    frame_length = 1024
    n_frames = len(f0)
    total_samples = (n_frames - 1) * hop_length + frame_length
    if total_samples > len(wav):
        wav_padded = np.pad(wav, (0, total_samples - len(wav)))
    else:
        wav_padded = wav[:total_samples]
    frames = librosa.util.frame(wav_padded, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    energy_db = 20 * np.log10(rms + 1e-8)
    energy_db = energy_db[:n_frames]
    return f0, energy_db

total_pitch_rmse = 0
total_energy_rmse = 0
count = 0

gen_files = sorted(os.listdir(generated_dir))
print(f"Found {len(gen_files)} generated files")

for fname in tqdm(gen_files):
    if not fname.endswith('.wav'):
        continue
    gen_path = os.path.join(generated_dir, fname)
    orig_path = os.path.join(original_wav_dir, fname)

    if not os.path.exists(orig_path):
        continue

    gen_pitch, gen_energy = extract_pitch_and_energy(gen_path)
    orig_pitch, orig_energy = extract_pitch_and_energy(orig_path)

    # Trim to the same length (shorter one)
    min_len = min(len(gen_pitch), len(orig_pitch))
    gen_pitch = gen_pitch[:min_len]
    orig_pitch = orig_pitch[:min_len]
    gen_energy = gen_energy[:min_len]
    orig_energy = orig_energy[:min_len]

    # Mask where both have valid pitch (>0)
    mask = (orig_pitch > 0) & (gen_pitch > 0)
    if mask.sum() < 3:
        continue

    pitch_rmse = np.sqrt(np.mean((gen_pitch[mask] - orig_pitch[mask])**2))
    energy_rmse = np.sqrt(np.mean((gen_energy[mask] - orig_energy[mask])**2))

    total_pitch_rmse += pitch_rmse
    total_energy_rmse += energy_rmse
    count += 1

if count > 0:
    avg_pitch_rmse = total_pitch_rmse / count
    avg_energy_rmse = total_energy_rmse / count
    print(f"\nProcessed {count} files.")
    print(f"Average Pitch RMSE: {avg_pitch_rmse:.2f} Hz")
    print(f"Average Energy RMSE: {avg_energy_rmse:.2f} dB")
else:
    print("No valid files processed.")
