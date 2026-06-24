import streamlit as st
import torch
import yaml
import json
import os
import glob
import time
import tempfile
import numpy as np
import librosa
import pyworld as pw
from g2p_en import G2p
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from streamlit_drawable_canvas import st_canvas
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from drawspeech.utilities.model_util import instantiate_from_config
from drawspeech.utilities.data.dataset import AudioDataset

# -------------------- 1. Load model --------------------
@st.cache_resource
def load_model(config_yaml_path, checkpoint_path):
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)
    config["reload_from_ckpt"] = checkpoint_path
    seed_everything(0)

    latent_diffusion = instantiate_from_config(config["model"])
    latent_diffusion.set_log_dir(config["log_directory"], "streamlit", "demo")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    latent_diffusion.load_state_dict(state_dict, strict=True)

    for key in latent_diffusion.cond_stage_model_metadata.keys():
        model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
        model = latent_diffusion.cond_stage_models[model_idx]
        if hasattr(model, 'infer'):
            model.infer = True

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()
    return latent_diffusion, config


# -------------------- 2. Synthesis --------------------
def synthesize(model, config, text, pitch_sketch, energy_sketch, duration=None):
    g2p = G2p()
    phones = g2p(text)
    phoneme_str = "{" + " ".join(phones) + "}"
    n_phones = len(phones)

    def pad_sketch(sketch):
        if sketch is None:
            return None
        arr = np.array(sketch)
        if len(arr) < n_phones:
            arr = np.pad(arr, (0, n_phones - len(arr)), constant_values=0.5)
        else:
            arr = arr[:n_phones]
        return arr.tolist()

    pitch_sketch = pad_sketch(pitch_sketch)
    energy_sketch = pad_sketch(energy_sketch)
    if duration is None:
        duration = [10] * n_phones

    def save_temp(arr):
        if arr is None:
            return ""
        path = tempfile.mktemp(suffix='.npy')
        np.save(path, np.array(arr))
        return path

    pitch_path = save_temp(pitch_sketch)
    energy_path = save_temp(energy_sketch)

    sample = {
        "wav": "dummy.wav",
        "speaker": "LJSpeech",
        "transcription": text,
        "phonemes": phoneme_str,
        "pitch_sketch": pitch_path,
        "energy_sketch": energy_path,
        "pitch": "",
        "energy": "",
        "duration": duration,
        "pitch_length": n_phones if pitch_sketch is not None else 0,
        "energy_length": n_phones if energy_sketch is not None else 0
    }
    dataset_json = {"data": [sample]}

    config["preprocessing"]["preprocessed_data"] = {
        "pitch": "data/dataset/metadata/ljspeech/phoneme_level/pitch",
        "energy": "data/dataset/metadata/ljspeech/phoneme_level/energy",
        "duration": "data/dataset/metadata/ljspeech/phoneme_level/duration",
        "stats_json": "data/dataset/metadata/ljspeech/phoneme_level/stats.json",
        "feature": "phoneme_level"
    }

    dataloader_add_ons = config["data"].get("dataloader_add_ons", [])
    dataset = AudioDataset(config=config, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json)
    loader = DataLoader(dataset, batch_size=1)

    eval_params = config["model"]["params"]["evaluation_params"]
    guidance_scale = eval_params["unconditional_guidance_scale"]
    ddim_steps = eval_params["ddim_sampling_steps"]
    n_gen = eval_params["n_candidates_per_samples"]

    with torch.no_grad():
        model.generate_sample(
            loader,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_gen
        )

    demo_dir = os.path.join(config["log_directory"], "streamlit", "demo")
    infer_dirs = sorted(glob.glob(os.path.join(demo_dir, "infer_*")), key=os.path.getmtime)
    if infer_dirs:
        latest_dir = infer_dirs[-1]
        wav_files = sorted(glob.glob(os.path.join(latest_dir, "*.wav")), key=os.path.getmtime)
        if wav_files:
            return wav_files[-1], pitch_sketch, energy_sketch
    return None, pitch_sketch, energy_sketch


# -------------------- 3. Drawing → sketch array --------------------
def drawing_to_sketch(canvas_result, n_phones):
    if canvas_result is None or canvas_result.json_data is None:
        return [0.5] * n_phones

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return [0.5] * n_phones

    points = []
    for obj in objects:
        if obj.get("type") == "path" and "path" in obj:
            for seg in obj["path"]:
                if len(seg) == 3:
                    points.append((seg[1], seg[2]))
    if not points:
        return [0.5] * n_phones

    points.sort(key=lambda p: p[0])
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    canvas_height = 200
    if canvas_result.json_data.get("height"):
        canvas_height = canvas_result.json_data["height"]
    ys = 1.0 - (ys / canvas_height)          # invert so top = 1
    ys = np.clip(ys, 0.0, 1.0)

    x_new = np.linspace(min(xs), max(xs), n_phones)
    f = interp1d(xs, ys, kind='linear', fill_value="extrapolate")
    sketch = f(x_new).clip(0.0, 1.0).tolist()
    return sketch


# -------------------- 4. Pitch extraction from generated WAV --------------------
def extract_pitch_from_wav(wav_path, n_phones):
    wav, sr = librosa.load(wav_path, sr=22050)
    wav = wav.astype(np.float64)
    hop_length = 256
    f0, t = pw.dio(wav, sr, frame_period=hop_length/sr*1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    nonzero = np.where(f0 > 0)[0]
    if len(nonzero) < 2:
        return None
    f0_interp = np.interp(np.arange(len(f0)), nonzero, f0[nonzero])
    x_old = np.linspace(0, 1, len(f0_interp))
    x_new = np.linspace(0, 1, n_phones)
    f0_phoneme = np.interp(x_new, x_old, f0_interp)
    f0_min, f0_max = f0_phoneme.min(), f0_phoneme.max()
    if f0_max - f0_min < 1e-6:
        return None
    return (f0_phoneme - f0_min) / (f0_max - f0_min)


# -------------------- 5. UI --------------------
def main():
    st.set_page_config(page_title="DrawSpeech Demo", layout="wide")
    st.title("DrawSpeech – Expressive Speech Synthesis with Sketches")
    st.markdown("Draw a pitch contour **or** use sliders (word / phoneme level) to control the intonation.")

    with st.spinner("Loading DrawSpeech model..."):
        model, config = load_model(
            "drawspeech/config/drawspeech_ljspeech_22k.yaml",
            "data/checkpoints/drawspeech_fixed.ckpt"
        )
    st.success("Model ready!")

    text = st.text_input("Enter text:", "I didn't say you stole the money.")
    g2p = G2p()
    phones = g2p(text)
    n_phones = len(phones)

    words = text.split()
    word_phones = [len(g2p(word)) for word in words]
    if sum(word_phones) != n_phones:
        word_phones = [n_phones // len(words)] * len(words)
        word_phones[-1] += n_phones - sum(word_phones)

    st.write(f"**Words:** {' | '.join(words)}")
    st.write(f"**Phonemes per word:** {dict(zip(words, word_phones))}")

    # ---- Choose input mode ----
    input_mode = st.radio(
        "Input mode:",
        ["Sketch", "Word sliders", "Phoneme sliders"],
        horizontal=True
    )

    pitch_sketch = None

        # ---- Choose input mode ----
    input_mode = st.radio(
        "Input mode:",
        ["Word Sketch", "Word Curve", "Phoneme Sketch", "Phoneme Curve"],
        horizontal=True
    )

    pitch_sketch = None
    canvas_width = 600   # default for word-level sketches

    if input_mode == "Word Sketch":
        st.subheader("Pitch Sketch (over words)")
        st.caption("Draw a line from left to right. Top = higher pitch, Bottom = lower pitch.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=2,
            stroke_color="#0000FF",
            height=200,
            width=canvas_width,
            drawing_mode="freedraw",
            key="pitch_word_sketch",
            update_streamlit=True,
        )

        # Word‑aligned x‑axis guide
        total_phones = sum(word_phones)
        guide_html = f'<div style="display:flex; width:{canvas_width}px; margin-top:5px;">'
        for i, (word, count) in enumerate(zip(words, word_phones)):
            width_pct = (count / total_phones) * 100
            border_style = "border-left: 1px dashed gray;" if i > 0 else ""
            guide_html += f'<div style="flex-basis:{width_pct}%; text-align:center; font-size:12px; {border_style}">{word}</div>'
        guide_html += '</div>'
        st.markdown(guide_html, unsafe_allow_html=True)

        if canvas_result.json_data is not None:
            pitch_sketch = drawing_to_sketch(canvas_result, n_phones)

    elif input_mode == "Word Curve":
        st.subheader("Word‑level sliders (displayed as a curve)")
        st.caption("Set a pitch value for each word. Values are shown as a connected line.")

        word_values = []
        cols = st.columns(len(words))
        for word, col in zip(words, cols):
            with col:
                val = st.slider(word, 0.0, 1.0, 0.5, key=f"wc_{word}")
                word_values.append(val)

        if word_values:
            # Expand word values to phoneme length
            pitch_sketch = []
            for val, count in zip(word_values, word_phones):
                pitch_sketch.extend([val] * count)

            # Show a small line plot of the word values
            fig_word, ax_word = plt.subplots(figsize=(4, 1.5))
            ax_word.plot(word_values, marker='o')
            ax_word.set_xticks(range(len(words)))
            ax_word.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
            ax_word.set_ylim(0, 1)
            ax_word.set_ylabel('Pitch')
            st.pyplot(fig_word)

    elif input_mode == "Phoneme Sketch":
        st.subheader("Pitch Sketch (over phonemes)")
        st.caption("Draw a line from left to right. Each phoneme is labelled below.")

        # Wider canvas for many phonemes – up to 1200px, but limited by screen
        phoneme_canvas_width = min(1200, max(600, n_phones * 20))
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=2,
            stroke_color="#0000FF",
            height=200,
            width=phoneme_canvas_width,
            drawing_mode="freedraw",
            key="pitch_phoneme_sketch",
            update_streamlit=True,
        )

        # Phoneme labels as a horizontal scrollable row (if too wide)
        label_html = '<div style="overflow-x: auto; white-space: nowrap; margin-top:5px;">'
        for i, phone in enumerate(phones):
            label_html += f'<span style="display:inline-block; width:{max(20, phoneme_canvas_width//n_phones)}px; text-align:center; font-size:10px; border-right:1px solid #ddd;">{phone}</span>'
        label_html += '</div>'
        st.markdown(label_html, unsafe_allow_html=True)

        if canvas_result.json_data is not None:
            pitch_sketch = drawing_to_sketch(canvas_result, n_phones)

    elif input_mode == "Phoneme Curve":
        st.subheader("Phoneme‑level sliders (displayed as a curve)")
        st.caption("Set a pitch value for each phoneme.")

        pitch_sketch = []
        cols = st.columns(min(10, n_phones))
        col_idx = 0
        for i, phone in enumerate(phones):
            if col_idx >= 10:
                cols = st.columns(min(10, n_phones - i))
                col_idx = 0
            with cols[col_idx]:
                val = st.slider(phone, 0.0, 1.0, 0.5, key=f"pc_{i}")
                pitch_sketch.append(val)
            col_idx += 1

        # Show a line plot of the phoneme values (optional, but nice)
        if pitch_sketch:
            fig_phon, ax_phon = plt.subplots(figsize=(6, 1.5))
            ax_phon.plot(pitch_sketch)
            ax_phon.set_ylim(0, 1)
            ax_phon.set_ylabel('Pitch')
            ax_phon.set_xlabel('Phoneme index')
            st.pyplot(fig_phon)

    # Neutral energy for now
    energy_sketch = [0.5] * n_phones


    
    if st.button("Generate Speech"):
        if pitch_sketch is None:
            pitch_sketch = [0.5] * n_phones

        with st.spinner("Synthesizing..."):
            audio_path, used_pitch, used_energy = synthesize(
                model, config, text, pitch_sketch, energy_sketch
            )
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
                st.success("Done!")
                with open(audio_path, "rb") as f:
                    st.download_button("Download WAV", f, file_name="drawspeech_output.wav")

                st.subheader("Sketch vs. Generated Pitch")
                gen_pitch = extract_pitch_from_wav(audio_path, n_phones)
                if gen_pitch is not None:
                    sketch_arr = np.array(used_pitch)
                    gen_arr = np.array(gen_pitch)
                    corr = np.corrcoef(sketch_arr, gen_arr)[0,1]
                    rmse = np.sqrt(np.mean((sketch_arr - gen_arr)**2))
                    st.metric("Correlation (Pearson's r)", f"{corr:.3f}")
                    st.metric("RMSE (normalized)", f"{rmse:.4f}")

                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(sketch_arr, label="Sketch", marker="o")
                    ax.plot(gen_arr, label="Generated pitch (normalized)", marker="x")
                    ax.set_xlabel("Phoneme index")
                    ax.set_ylabel("Normalized pitch")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("Could not extract pitch from generated audio.")
            else:
                st.error("Generation failed. Check terminal for details.")


if __name__ == "__main__":
    main()