# drawspeech_addon
Write a thorough guide (you can base it on the report I just gave you). Here’s a skeleton you can copy and edit:

bash
nano README.md
Paste:

markdown
# DrawSpeech Reproduction with Interactive Sketch Control

This repository contains a reproducible version of [DrawSpeech](https://github.com/HappyColor/DrawSpeech_PyTorch) with fixes for modern libraries, plus an interactive **Streamlit** demo for controlling expressive speech synthesis using drawn pitch sketches, word‑level sliders, or phoneme‑level sliders.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/neelima16/drawspeech_addon.git
cd drawspeech_addon
2. Create conda environment and install dependencies
bash
conda create -n drawspeech python=3.10
conda activate drawspeech
pip install -r requirements.txt
3. Download the LJSpeech dataset and alignments
LJSpeech: download from here and extract into data/dataset/LJSpeech-1.1/.

TextGrid alignments: download from the original repository and place them in data/dataset/LJSpeech-1.1/TextGrid/LJSpeech/.

4. Download checkpoints and vocoder
VAE checkpoint (vae.ckpt) and DrawSpeech checkpoint (drawspeech.ckpt) from the official HuggingFace page → place them in data/checkpoints/.

HiFi‑GAN vocoder: download config.json and the pretrained generator from here into data/checkpoints/LJ_V1/. Rename the generator file to generator_v1.

5. Fix checkpoint key names
The provided drawspeech.ckpt uses old naming conventions. Run:

bash
python fix_checkpoint.py
This creates data/checkpoints/drawspeech_fixed.ckpt with corrected keys.

6. Preprocess the dataset
bash
python preprocessing.py
This step extracts pitch, energy, duration, and saves phoneme‑level features.

7. (Optional) Batch inference on test set
Create tests/inference.json by adding the paths to the preprocessed features to the test split JSON. Then run:

bash
python drawspeech/infer.py \
  --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
  --list_inference tests/inference.json \
  --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
This generates 300 test utterances. To compute RMSE against the original LJSpeech, run:

bash
python compute_rmse.py
8. Interactive Demo
Launch the Streamlit app:

bash
streamlit run app.py --server.port 8501
On your local machine, create an SSH tunnel to the GPU node (e.g., ssh -L 8502:localhost:8501 user@login_node), then open http://localhost:8502. You can choose between Word Sketch, Word Curve, Phoneme Sketch, and Phoneme Curve input modes. After generating speech, a plot comparing your sketch to the extracted pitch and the correlation/RMSE are displayed.

Modifications made to the original code
(Summarise the changes, e.g., stft.py, preprocessor.py, conditional_models.py, ddpm.py, infer.py, config YAML, etc.)

Troubleshooting
ModuleNotFoundError: pkg_resources → pip install setuptools==68.2.2

AttributeError: 'NoneType' object has no attribute 'bool' → See conditional_models.py fix

If the app can't find taming, ensure taming-transformers is installed and in PYTHONPATH.

text

### 4. Add the modified files

Make sure you've added all the modified files to the repository. If you cloned the official repo and then made changes, those changes are already tracked by Git (if you committed them). If not, do:

```bash
git add -A
git commit -m "Add modifications for reproducibility, Streamlit app, RMSE script, and documentation"
