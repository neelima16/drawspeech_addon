import os, argparse, numpy as np, librosa, pyworld as pw, tqdm

def extract_pitch_and_energy(wav_path, sr=22050, hop_length=256):
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=hop_length/sr*1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    frame_length = 1024
    n_frames = len(f0)
    total_samples = (n_frames - 1) * hop_length + frame_length
    wav_pad = np.pad(wav, (0, max(0, total_samples - len(wav))))[:total_samples]
    frames = librosa.util.frame(wav_pad, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    energy_db = 20 * np.log10(rms + 1e-8)
    energy_db = energy_db[:n_frames]
    return f0, energy_db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', required=True)
    parser.add_argument('--original_dir', required=True)
    args = parser.parse_args()

    gen_files = sorted(os.listdir(args.generated_dir))
    total_pitch, total_energy, count = 0, 0, 0
    for fname in tqdm.tqdm(gen_files):
        if not fname.endswith('.wav'):
            continue
        gen_path = os.path.join(args.generated_dir, fname)
        orig_path = os.path.join(args.original_dir, fname)
        if not os.path.exists(orig_path):
            continue
        gen_pitch, gen_energy = extract_pitch_and_energy(gen_path)
        orig_pitch, orig_energy = extract_pitch_and_energy(orig_path)
        min_len = min(len(gen_pitch), len(orig_pitch))
        gen_pitch, orig_pitch = gen_pitch[:min_len], orig_pitch[:min_len]
        gen_energy, orig_energy = gen_energy[:min_len], orig_energy[:min_len]
        mask = (orig_pitch > 0) & (gen_pitch > 0)
        if mask.sum() < 3:
            continue
        pitch_rmse = np.sqrt(np.mean((gen_pitch[mask] - orig_pitch[mask])**2))
        energy_rmse = np.sqrt(np.mean((gen_energy[mask] - orig_energy[mask])**2))
        total_pitch += pitch_rmse
        total_energy += energy_rmse
        count += 1
    if count:
        print(f"Processed {count} files")
        print(f"Average Pitch RMSE: {total_pitch/count:.2f} Hz")
        print(f"Average Energy RMSE: {total_energy/count:.2f} dB")
    else:
        print("No valid files processed.")

if __name__ == '__main__':
    main()
