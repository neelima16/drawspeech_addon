<<<<<<< HEAD

# DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions
ICASSP 2025 [[Paper]](https://ieeexplore.ieee.org/abstract/document/10889767) [[arXiv]](https://arxiv.org/abs/2501.04256) [[Demo]](https://happycolor.github.io/DrawSpeech/)

## Status
This project is currently under active development. We are continuously updating and improving it, with more usage details and features to be released in the future.

# Getting started

## Download dataset and checkpoints 
1. Download the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset and place the dataset into `data/dataset` with structure looks like below:
```plaintext
data/dataset/LJSpeech-1.1
 ┣ metadata.csv
 ┣ wavs
 ┃ ┣ LJ001-0001.wav
 ┃ ┣ LJ001-0002.wav 
 ┃ ┣ ...
 ┣ README
```
2. Download the alignments of the LJSpeech dataset [LJSpeech.zip](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4). You have to unzip the files in `data/dataset/LJSpeech-1.1`
3. Download checkpoints from [HuggingFace](https://huggingface.co/HappyColor/DrawSpeech/tree/main).
4. Place the checkpoints into **data/checkpoints/**

## Preprocessing
```python
python preprocessing.py
```

## Training

Train the VAE (Optional)
```python
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/autoencoder.py -c drawspeech/config/vae_ljspeech_22k.yaml
```

If you don't want to train the VAE, you can just use the VAE checkpoint that we provide.
- set the variable `reload_from_ckpt` in `drawspeech_ljspeech_22k.yaml` to `data/checkpoints/vae.ckpt`

Train the DrawSpeech
```python
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py -c drawspeech/config/drawspeech_ljspeech_22k.yaml
```


## Inference

If you have trained the model using `drawspeech_ljspeech_22k.yaml`, use the following syntax:
```shell
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/inference.json
```

If not, please specify the DrawSpeech checkpoint:
```shell
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/inference.json --reload_from_ckpt data/checkpoints/drawspeech.ckpt
```

## Acknowledgement
This repository borrows codes from the following repos. Many thanks to the authors for their great work.
* [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning?tab=readme-ov-file#prepare-python-running-environment)
* [FastSpeech 2](https://github.com/ming024/FastSpeech2) 
* [HiFi-GAN](https://github.com/jik876/hifi-gan)

## Citation
```bibtex
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889767}}
}
```
=======
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
>>>>>>> origin/main
