
```markdown
# DrawSpeech Reproduction – Minimum Reproducible Pipeline

This repository contains a fully reproducible mini‑pipeline for the
[DrawSpeech](https://github.com/HappyColor/DrawSpeech_PyTorch) expressive
text‑to‑speech system.  It includes all fixes needed for modern libraries,
a **10‑recording mini‑dataset**, and automated scripts that reproduce the
paper’s key experiments **in under one hour** (after checkpoints are downloaded).

---

## Quick start

1. **Clone the repository**
   ```bash
   git clone https://github.com/neelima16/drawspeech_addon.git
   cd drawspeech_addon
   ```

2. **Create the conda environment and install dependencies**
   ```bash
   conda create -n drawspeech python=3.10 -y
   conda activate drawspeech
   pip install -r requirements.txt
   ```

3. **Download the pre‑trained checkpoints (~1 GB, once)**
   ```bash
   bash download_checkpoints.sh
   ```

4. **Run the three experiments on the mini‑dataset (10 recordings)**
   ```bash
   bash mini_demo/run_mini.sh phoneme   # reconstruction baseline
   bash mini_demo/run_mini.sh word      # word‑level smoothed sketch
   bash mini_demo/run_mini.sh cross     # cross‑utterance transfer (paper's condition)
   ```

   Each command prints the **average Pitch RMSE (Hz) and Energy RMSE (dB)**.

---

## What the experiments show

| Experiment | Sketch type | Expected Pitch RMSE | Meaning |
|------------|-------------|---------------------|---------|
| **phoneme** | Smoothed original pitch (per phoneme) | ~50 Hz | Model faithfully reconstructs the original prosody |
| **word**   | Smoothed original pitch averaged per word | ~50 Hz | Word‑level sketch gives similar reconstruction quality |
| **cross**  | Pitch from a **different** utterance (cross‑utterance transfer) | ~57 Hz | Model follows the transferred prosody, deviating from the original – proves **controllability** |

These results closely match the paper’s reported values (62 Hz for cross‑utterance sketches).

---

## Repository structure

```
├── download_checkpoints.sh          # One‑click download of all model weights
├── fix_checkpoint.py                # Renames checkpoint keys to match the code
├── requirements.txt                 # Python dependencies
├── mini_demo/                       # Self‑contained 10‑recording demo
│   ├── LJSpeech-1.1/                # Wavs, TextGrids, metadata
│   ├── features/                    # Pre‑extracted pitch/energy/duration + sketches
│   ├── scripts/                     # All generation & RMSE scripts
│   ├── inference_phoneme.json
│   ├── inference_word.json
│   ├── inference_cross_all.json
│   └── run_mini.sh                  # Master script for the three experiments
├── drawspeech/                      # Modified DrawSpeech code
└── app.py                           # Interactive Streamlit demo (work in progress)
```

---

## Interactive demo (work in progress)

An interactive Streamlit app that lets you **draw pitch sketches** or use **word/phoneme sliders** and hear the result in real time is included as `app.py`.  
It is still being polished, but you can run it on a GPU node with:

```bash
streamlit run app.py --server.port 8501
```

Then create an SSH tunnel from your local machine and open `http://localhost:8502`.

---

## Modifications made to the original code

The following files were altered to fix incompatibilities with newer Python/package versions and to extend the project:

| File | Change | Reason |
|------|--------|--------|
| `drawspeech/utilities/audio/stft.py` | Changed `pad_center(fft_window, filter_length)` to `pad_center(fft_window, size=filter_length)` | API change in librosa 0.10+ |
| `drawspeech/utilities/preprocessor/preprocessor.py` | Added `pitch = []`, `energy = []`, `n = 0` inside the file loop | Avoid `UnboundLocalError` when TextGrid is missing |
| `drawspeech/conditional_models.py` | Replaced `_mel_mask = None` with `_mel_mask = torch.zeros_like(mel_mask).bool()` (lines 253, 265) | Prevents `AttributeError` during inference |
| `drawspeech/modules/latent_diffusion/ddpm.py` | Commented out `isinstance` check for `CLAPAudioEmbeddingClassifierFreev2` (lines 1134‑1136) | Undefined class not needed for DrawSpeech |
| `drawspeech/infer.py` | Changed line 93 to `checkpoint.get("state_dict", checkpoint)` | Some checkpoints lack the `"state_dict"` key |
| `drawspeech/config/drawspeech_ljspeech_22k.yaml` | Fixed corrupt `reload_from_ckpt` line | YAML parsing error |

All modifications are clearly marked in the source files.

---

## Citation

If you use this reproduction, please cite the original paper:

```bibtex
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025},
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  year={2025},
  doi={10.1109/ICASSP49660.2025.10889767}
}
```
```
