
# DrawSpeech Reproduction with Interactive Sketch Control

This repository contains a fully reproducible version of [DrawSpeech](https://github.com/HappyColor/DrawSpeech_PyTorch) – a sketch‑conditioned diffusion model for expressive text‑to‑speech synthesis. It includes **all required fixes** for modern library versions, a **Streamlit interactive demo** where you can draw pitch sketches or use word/phoneme sliders, and a script to compute **RMSE** between generated and original speech.

---

## Table of Contents
- [Setup](#setup)
- [Data & Checkpoints](#data--checkpoints)
- [Preprocessing](#preprocessing)
- [Fix Checkpoint Keys](#fix-checkpoint-keys)
- [Batch Inference (Test Set)](#batch-inference-test-set)
- [RMSE Calculation](#rmse-calculation)
- [Interactive Demo](#interactive-demo)
- [Modifications Made to the Original Code](#modifications-made-to-the-original-code)
- [Troubleshooting](#troubleshooting)
- [Results](#results)

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/neelima16/drawspeech_addon.git
cd drawspeech_addon
```

### 2. Create and activate a conda environment
```bash
conda create -n drawspeech python=3.10
conda activate drawspeech
```

### 3. Install dependencies
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

> **Note**: If you encounter a `ModuleNotFoundError: pkg_resources`, downgrade `setuptools`:
> ```bash
> pip install setuptools==68.2.2
> ```

---

## Data & Checkpoints

### 1. LJSpeech dataset
Download the [LJSpeech dataset](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) (~2.6 GB) and extract it so that the folder structure looks like:
```
data/dataset/LJSpeech-1.1/
├── metadata.csv
├── wavs/
│   ├── LJ001-0001.wav
│   └── ...
└── README
```

### 2. TextGrid Alignments
Download the alignments from the [original DrawSpeech repository](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4) (or use pre‑computed MFA alignments). Place them inside a `TextGrid/LJSpeech/` subfolder:
```
data/dataset/LJSpeech-1.1/TextGrid/LJSpeech/
├── LJ001-0001.TextGrid
├── LJ001-0002.TextGrid
└── ...
```

> **Important**: The code expects the speaker folder `LJSpeech` inside `TextGrid/`. If your extracted files are directly in `TextGrid/`, create the subfolder and move them there.

### 3. Pre‑trained Checkpoints
Download the official checkpoints from [HuggingFace](https://huggingface.co/HappyColor/DrawSpeech/tree/main):
- `vae.ckpt`
- `drawspeech.ckpt`

Place them into `data/checkpoints/`.

### 4. HiFi‑GAN Vocoder
The vocoder converts mel‑spectrograms to waveforms. Download:
- `config.json` from [here](https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json)
- Pre‑trained generator from [here](https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_LJSpeech.pth.tar) (or via Google Drive if the link is blocked).

Place them in `data/checkpoints/LJ_V1/`. **Rename** the generator file to `generator_v1` (remove `.pth` or `.tar` extension):
```bash
mkdir -p data/checkpoints/LJ_V1
mv generator_LJSpeech.pth.tar data/checkpoints/LJ_V1/generator_v1
```

If you encounter a 404 or proxy issues, use the curl command with `-L` or download on the login node.

---

## Preprocessing

The preprocessing step extracts phoneme‑level pitch, energy, duration, and normalisation statistics. It also creates train/val/test splits.

```bash
python preprocessing.py
```

This will take about 30 minutes (depending on your hardware) and will generate the directory `data/dataset/metadata/ljspeech/phoneme_level/` containing the `.npy` feature files and `stats.json`.

---

## Fix Checkpoint Keys

The provided `drawspeech.ckpt` uses an older naming convention. The current code expects slightly different parameter names. Run the included script to create a corrected checkpoint:

```bash
python fix_checkpoint.py
```

This will create `data/checkpoints/drawspeech_fixed.ckpt`. From now on, always use this fixed checkpoint for inference and the demo.

---

## Batch Inference (Test Set)

You can generate speech for the entire LJSpeech test set (300 utterances) using the original pitch/energy contours (not sketches). This verifies that the model works correctly.

### 1. Create the inference JSON file
The inference script expects a JSON file that lists the test samples **and** the paths to the pre‑processed features. Run this one‑liner:

```bash
python -c "
import json
with open('data/dataset/metadata/ljspeech/test.json') as f:
    test = json.load(f)
test['pitch'] = 'data/dataset/metadata/ljspeech/phoneme_level/pitch'
test['energy'] = 'data/dataset/metadata/ljspeech/phoneme_level/energy'
test['duration'] = 'data/dataset/metadata/ljspeech/phoneme_level/duration'
with open('tests/inference.json', 'w') as f:
    json.dump(test, f, indent=2)
"
```

### 2. Run inference on a GPU node
```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
python drawspeech/infer.py \
  --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
  --list_inference tests/inference.json \
  --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
```

The generated WAV files will be saved under `log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_<timestamp>_.../`.

> **Note**: If you don't have `taming-transformers`, see [Troubleshooting](#troubleshooting).

---

## RMSE Calculation

To quantitatively assess how well the model reproduces the original test utterances, use the provided script:

```bash
python compute_rmse.py
```

It will process all 300 files (takes a few minutes) and print the **average Pitch RMSE (Hz)** and **average Energy RMSE (dB)**.

*Expected result when using ground‑truth pitch/energy:*  
Pitch RMSE ~ 22–25 Hz, Energy RMSE ~ 3–4 dB.  
These values confirm near‑perfect reconstruction.

---

## Interactive Demo

A Streamlit app lets you control the prosody in **real time** by drawing a pitch sketch or using word/phoneme sliders.

### 1. Start the app on a GPU node
```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
streamlit run app.py --server.port 8501
```

### 2. Connect from your local machine
Create an SSH tunnel to the GPU node (replace `tinyx` with your login node, and `tg071` with the compute node you are on):

```bash
ssh -L 8502:tg071:8501 your_user@tinyx
```

Then open `http://localhost:8502` in your browser.

### 3. Using the demo
- **Input text**: Type any sentence (default: “I didn't say you stole the money.”).
- **Input mode**: Choose between:
  - **Word Sketch** – draw a pitch curve over word‑aligned guides.
  - **Word Curve** – sliders for each word, displayed as a curve.
  - **Phoneme Sketch** – draw over phoneme labels (canvas widens for long sentences).
  - **Phoneme Curve** – sliders for each phoneme, shown as a curve.
- Click **Generate Speech**. The model synthesises the audio and plays it in the browser.
- After generation, a plot comparing your sketch (or slider values) to the **extracted pitch** from the output is shown, along with **Pearson correlation** and **normalised RMSE**. A download button lets you save the WAV file.

---

## Modifications Made to the Original Code

The following files were altered to fix incompatibilities with newer Python/package versions and to add the interactive demo:

| File | Change | Reason |
|------|--------|--------|
| `drawspeech/utilities/audio/stft.py` | Changed `pad_center(fft_window, filter_length)` to `pad_center(fft_window, size=filter_length)` | API change in librosa 0.10+ |
| `drawspeech/utilities/preprocessor/preprocessor.py` | Added `pitch = []`, `energy = []`, `n = 0` inside the file loop | Avoid `UnboundLocalError` when TextGrid is missing |
| `drawspeech/conditional_models.py` | Replaced `_mel_mask = None` with `_mel_mask = torch.zeros_like(mel_mask).bool()` (lines 253, 265) | Prevents `AttributeError` during inference |
| `drawspeech/modules/latent_diffusion/ddpm.py` | Commented out `isinstance` check for `CLAPAudioEmbeddingClassifierFreev2` (lines 1134‑1136) | Undefined class not needed for DrawSpeech |
| `drawspeech/infer.py` | Changed line 93 to `checkpoint.get("state_dict", checkpoint)` | Some checkpoints lack the `"state_dict"` key |
| `drawspeech/config/drawspeech_ljspeech_22k.yaml` | Fixed corrupt `reload_from_ckpt` line | YAML parsing error |
| `app.py` (new) | Complete Streamlit demo with sketch/slider modes, RMSE comparison | Interactive prosody control |
| `compute_rmse.py` (new) | Script to compute pitch/energy RMSE on test set | Quantitative evaluation |
| `fix_checkpoint.py` (new) | Renames old checkpoint keys to match current code | Avoids key mismatch errors |

> All original functionality is preserved. The changes only ensure compatibility and extend the project.

---

## Troubleshooting

### `ModuleNotFoundError: pkg_resources`
```bash
pip install setuptools==68.2.2
```

### `ModuleNotFoundError: taming` or `einops`
- **taming**: Clone `taming-transformers` into the project root and install:
  ```bash
  git clone https://github.com/CompVis/taming-transformers.git
  cd taming-transformers
  pip install -e .
  cd ..
  ```
  Also add it to `PYTHONPATH` as shown above.
- **einops**: `pip install einops`

### `AttributeError: 'NoneType' object has no attribute 'bool'`
Make sure you’ve applied the fix to `conditional_models.py` (line 253, 265). If the error still appears, restart the app.

### `KeyError: 'stats_json'` or `'feature'`
The demo app automatically sets these config values. If you run inference manually, ensure your JSON includes the paths to the preprocessed data and that the config has the correct `preprocessed_data` dictionary.

### GPU memory errors
Reduce `batch_size` in the config YAML or use a smaller test set for inference.

---

## Results

- **Preprocessing**: Successfully extracted features for all 13,100 utterances.
- **Test set inference**: 300 utterances generated with **Pitch RMSE = 22.65 Hz**, **Energy RMSE = 3.69 dB** (using ground‑truth prosody).
- **Interactive demo**: Users can draw any pitch contour and immediately hear the expressive speech; the on‑screen plot shows the correlation between the sketch and the output.

---

## Acknowledgements
This work is based on the original [DrawSpeech](https://github.com/HappyColor/DrawSpeech_PyTorch) paper and code. Many thanks to the authors for their excellent research.

## Citation
Please cite the original paper if you use this repository:
```bibtex
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025},
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  year={2025},
  doi={10.1109/ICASSP49660.2025.10889767}
}
```

---

**Happy sketching! 🎤✏️**

```bash
git add README.md
git commit -m "Replace README with detailed reproduction guide"
git push origin main
```

This README will allow anyone to replicate your entire project from scratch, including the interactive demo, with all the necessary fixes and commands clearly documented.
