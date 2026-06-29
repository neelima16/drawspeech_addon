
# DrawSpeech Reproduction – Minimum Reproducible Pipeline (USE HPC)

This repository contains a fully reproducible pipeline for the
[DrawSpeech](https://github.com/HappyColor/DrawSpeech_PyTorch) expressive
text‑to‑speech system. It includes all fixes needed for modern libraries,
a **10‑recording mini‑dataset** for quick experiments, and automated scripts
that reproduce the paper’s key results **in under one hour** (after checkpoints
are downloaded). A full‑scale reproduction (300 test files, cross‑utterance
evaluation) is also supported.

---

## Quick start (mini‑dataset, 10 recordings)

1. **Clone the repository**
   ```bash
   git clone https://github.com/neelima16/drawspeech_addon.git
   cd drawspeech_addon
   ```

2. **Create the conda environment and install dependencies**
   ```bash
   module load Python
   module avail conda
   conda create -n drawspeech python=3.10 -y
   conda activate drawspeech
   pip install -r requirements.txt
   ```

3. **Download the pre‑trained checkpoints (~1 GB, once)**
   ```bash
   bash download_checkpoints.sh
   ```

4. **Run the three experiments**
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

An interactive Streamlit app lets you **draw pitch sketches** or use **word/phoneme sliders** and hear the result in real time. The app is still being polished, but you can test it on a GPU node:

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then, from your **local machine**, create an SSH tunnel. Replace `<GPU_HOST>` with the actual hostname of the GPU node (e.g., `tg090`), and `<LOGIN_NODE>` with your cluster's login address (e.g., `tinyx` or `csnhr.nhr.fau.de`):

```bash
ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
```

For example, if your username is `iwi5408h` and the GPU node is `tg090`:

```bash
ssh -L 8501:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```

Now open your browser and go to **http://localhost:8501**.  
(If port 8501 is blocked locally, use `-L 8502:...` and open `http://localhost:8502`.)

---


For a Mac user
Start the Streamlit app on the GPU node (as shown in your README).

Open Terminal on your Mac.

Create the SSH tunnel (replace tg090 with the actual GPU hostname and csnhr.nhr.fau.de with your login node):
```
bash
ssh -L 8501:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```
If port 8501 is already in use on your Mac, use a different local port, e.g.:
```
bash
ssh -L 8502:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```
Keep the Terminal window open.

Open your browser and go to http://localhost:8501 (or http://localhost:8502 if you used the alternate port).
## Modifications made to the original code

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

## Complete Pipeline (full dataset reproduction)

If you want to reproduce the **full paper evaluation** on the entire LJSpeech test set (300 utterances), follow these steps.  
This requires the full LJSpeech dataset (~2.6 GB) and about **2–4 hours** of preprocessing / inference.

### 1. Download the full LJSpeech dataset
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2 -C data/dataset/
```

### 2. Download the TextGrid alignments
Obtain the alignments from the [DrawSpeech repository](https://github.com/HappyColor/DrawSpeech_PyTorch) and place them under:
```
data/dataset/LJSpeech-1.1/TextGrid/LJSpeech/
```

### 3. Preprocess the full dataset
This extracts pitch, energy, duration, and creates the phoneme‑level features (takes ~30 min on a CPU node):
```bash
python preprocessing.py
```
Make sure the config `drawspeech/utilities/preprocessor/preprocess_phoneme_level.yaml` points to the correct paths.

### 4. Download checkpoints (if not already done)
```bash
bash download_checkpoints.sh
```

### 5. Run full test‑set inference with **original pitch as sketch**
```bash
# First create the inference JSON (adds the feature paths to test.json)
python -c "
import json
with open('data/dataset/metadata/ljspeech/test.json') as f:
    data = json.load(f)
data['pitch'] = 'data/dataset/metadata/ljspeech/phoneme_level/pitch'
data['energy'] = 'data/dataset/metadata/ljspeech/phoneme_level/energy'
data['duration'] = 'data/dataset/metadata/ljspeech/phoneme_level/duration'
with open('tests/inference_test.json', 'w') as f:
    json.dump(data, f, indent=2)
"

# Run inference
python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_test.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
```

### 6. Compute RMSE for the full test set
```bash
LATEST_DIR=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
python compute_rmse.py --generated_dir "$LATEST_DIR" --original_dir data/dataset/LJSpeech-1.1/wavs
```

### 7. Run cross‑utterance evaluation (paper's sketch condition, 300 files)
Use the provided sbatch script `run_full_cross.sbatch` (after adjusting the GPU type and partition):
```bash
sbatch run_full_cross.sbatch
```
This will automatically generate cross‑utterance sketches, run inference, and print the average RMSE – matching the paper's 62 Hz result.

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
