# Bridging Diffusion Models and LDDMM for Diffeomorphic Registration

Implementation of my Bachelor's Thesis in Computer Science at the University of Zaragoza, developed in collaboration with the Instituto de Investigación en Ingeniería de Aragón (I3A) under the supervision of Prof. Mónica Hernández Giménez.

---

## Overview

3D deformable image registration with a conditional denoising diffusion model, built on **DiffuseMorph**. This code adds:

1. **LDDMM-based deformation:** the network predicts a momentum field, a stationary velocity field is obtained from it with a differential regularization operator, and the deformation is computed by scaling and squaring (7 steps by default). The loss includes terms for the velocity field and the Jacobian determinant.
2. **3D bidirectional Mamba-2 blocks:** Mamba-2 blocks (`mamba_ssm`) scan the 3D feature volume along its three axes in both directions, conditioned on the diffusion timestep with Adaptive Layer Normalization (AdaLN). They can be placed at different levels of the U-Net (see below).

---

## Repository Structure

```text
├── config/          # JSON configuration files for training and evaluation
├── core/            # Logging and image utilities
├── data/            # OASIS dataset loader
├── model/           # Diffusion, LDDMM integration, and Mamba-2 U-Net architectures
├── main_3D.py       # Main entry point for 3D training and testing
└── requirements.txt # Python dependencies
```

---

## Requirements

The code requires a CUDA-capable NVIDIA GPU. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` does **not** include `mamba_ssm`, which must be installed separately and built against your own CUDA and PyTorch setup (see the [mamba repository](https://github.com/state-spaces/mamba)). Without it, the model cannot be imported.

---

## Data

Experiments use 3D T1-weighted brain MRI volumes from the **[OASIS dataset](https://www.oasis-brains.org/)** as distributed in the *Learn2Reg challenge* (skull-stripped, intensity-normalized, and affinely aligned). The loader expects this folder layout:

```text
OASIS/
├── imagesTr/               # training volumes (*.nii.gz)
├── imagesTs/               # test volumes (*.nii.gz)
├── labelsTr/               # training segmentations (*.nii.gz)
├── labelsTs/               # test segmentations (*.mat, see below)
└── OASIS_dataset.json      # Learn2Reg task file with the registration pairs
```

- **Training:** all `*.nii.gz` files in `dataroot` are used whose index (the second field of the file name when split by `_`) is lower than `train_max_index`. Labels are read from the matching file in `labelsTr` (`.nii.gz`).
- **Test:** the pairs are read from the `"registration_test"` list in `pairs_json`; each entry has a `"fixed"` and a `"moving"` image (only the file name is used and resolved against `dataroot`). The segmentations are read from `labelsTs` as **MATLAB `.mat` files**, one per image with the same name (`.nii.gz` replaced by `.mat`), whose first variable is the label volume. The Learn2Reg test labels are distributed as `.nii.gz`, so they must be converted to `.mat` before evaluation (for example with `nibabel` to load them and `scipy.io.savemat` to save them).

---

## Configuration

Edit the JSON files in `config/` before running:

- `datasets.<phase>.dataroot`: path to `imagesTr/` (train) or `imagesTs/` (test).
- `datasets.<phase>.pairs_json`: path to `OASIS_dataset.json`.
- `datasets.train.train_max_index`: training volumes are those with an index below this value.
- `path.resume_state`: checkpoint prefix. In training, `null` starts from scratch and a prefix resumes training. In testing it is **required**: set it to the prefix of a trained checkpoint, without the `_gen_G.pth` suffix (the code loads `<prefix>_gen_G.pth`). If it is `null` in testing, no weights are loaded and the network runs with random weights.

Replace the placeholder paths (`/path/to/...`) in the provided files with your own.

---

## Mamba-2 Variants (Up-All, Up-High, UpDown-High)

Where the Mamba-2 blocks are placed is set in `model/diffusion_net_3D/unet.py`, not in the JSON files. In the `UNet` constructor, the set `mamba_levels` selects the decoder levels that use a Mamba-2 block, and the `use_mamba` flag of the encoder (down) blocks and of the middle blocks enables them there. The three variants of the thesis (Up-All, Up-High, UpDown-High) are obtained by changing these settings in `unet.py` and training with the same configuration file. The code as provided has Mamba-2 blocks only in the decoder, at levels `{0, 1}`.

---

## Usage

### Training

```bash
python main_3D.py -p train -c config/diffuseMorph_train_3D.json
```

### Evaluation

To evaluate a trained checkpoint on the test split (Dice, Jacobian determinant statistics and inference time per pair), set `path.resume_state` in the test configuration and run:

```bash
python main_3D.py -p test -c config/diffuseMorph_test_3D.json
```

Each run creates a new folder `experiments/<name>_<timestamp>/` containing the `log`, `tb_logger`, `results` and `checkpoint` folders named in the `path` section of the configuration. Test results (`.mat` files and example images) are written to `experiments/<name>_<timestamp>/results/`, and training checkpoints to `.../checkpoint/`. The `resume_state` path is used as given and is not placed inside this folder.

---

```bibtex
@thesis{naudin2026diffusion,
  title={Bridging Diffusion Models and LDDMM for Diffeomorphic Registration},  
  author={Naudín López, Hugo},
  school={Universidad de Zaragoza},
  year={2026},
  type={Bachelor's Thesis}
}
```
