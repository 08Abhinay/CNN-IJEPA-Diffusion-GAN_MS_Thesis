# IJEPA-Diffusion-GAN 🩺✨  
*A hybrid generative framework for anatomy‑aware, data‑efficient medical‑image synthesis*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Papers](https://img.shields.io/badge/paper-Purdue%20M.S.%20Thesis-orange)

> **TL;DR** IJEPA‑Diffusion‑GAN fuses self‑supervised **I‑JEPA** semantic embeddings with the fidelity of diffusion models **and** the speed of StyleGAN‑family adversarial refinements.  
> Compared with pixel‑space diffusion it **cuts GPU memory by ≈48 %** while matching or beating state‑of‑the‑art FID/KID on chest‑X‑ray & brain‑MRI benchmarks.

---

## 1 Why this repo exists 🤔  

Medical imaging suffers from a three‑way trilemma: **fidelity, diversity, efficiency**.  
Traditional GANs collapse; pure diffusion is compute‑hungry.  
Our thesis shows that injecting *semantic guidance* from I‑JEPA into a diffusion‑GAN pipeline offers a balanced remedy.

---

## 2 Key features 🚀  

| 🔑  | What it does | Where to look |
|-----|--------------|---------------|
| **JEPAFusionMapping** | AdaIN‑style latent injection of 2 048‑D I‑JEPA codes into StyleGAN2‑ADA stages | `models/ijepa_fusion.py` |
| **Diffusion‑StyleGAN2 backbone** | DDPM steps inside StyleGAN2 discriminator for multiscale denoising | `models/diff_sg2/` |
| **Config‑driven experiment grid** | 3 backbones × 3 conditioning schemes × 2 datasets, fully reproducible | `configs/*.yaml`, `scripts/train.sh` |
| **Metrics & dashboards** | FID‑50k, KID, IS, Precision/Recall, W&B logging | `metrics/`, W&B link |

---

## 3 Project tree 🌲  

```text
.
├── calc_metrics_pr.py
├── calc_metrics.py
├── calc_metrics.sh
├── dataset_tool.py
├── environment.yml
├── gen_images.py
├── gen_video.py
├── generate_samples.sh
├── ijepa-film-stylegan2_gpu_4.sh
├── ijepa-film-stylegan2_gpu_4_brain.sh
├── legacy.py
├── train.py
├── diffusion-projected-lossonly-gan/
├── diffusion-projected-ramp-gan/
├── diffusion-ramp-stylegan2/
├── ijepa-diffusion-lossonly-stylegan2/
├── ijepa-lossonly-stylegan2/
├── ijepa-ramp-stylegan2/
├── dnnlib/
├── metrics/
├── pg_modules/
├── torch_utils/
└── training/
    ├── dataset.py
    ├── ijepa_encoder.py
    ├── loss.py
    └── training_loop.py
```

---

## 4 Setup & training 🔧  

### 4.1 Local environment (optional)  
Use this for **dataset conversion**, **metric evaluation**, or lightweight debugging.

```bash
conda create -n ijepa_diffusion_gan python=3.10      # ❶ create env
conda activate ijepa_diffusion_gan
pip install -r requirements.txt                      # ❷ install deps

# download pretrained I‑JEPA backbone (~120 MB)
bash scripts/download_ijepa.sh                       # ❸
```

*Training the full model requires multi‑GPU – see next section.*

### 4.2 HPC (SLURM + Apptainer) training  
Each folder in `scripts/StyleGAN2/*` ships with a ready‑to‑submit SLURM file like below
(`ijepa_SG_Chest_rampGD_warmup_5.4_4gpu_sem_mixing_0.9_FusionAlpha_0.2.sh`):

```bash
#!/bin/bash

export PROJECT=~/StyleGAN2/stylegan2-ada-pytorch
export TMPDIR=$PROJECT/tmp/torch_tmp
mkdir -p "$TMPDIR"

SEM_MIX=0.9
FUSION_ALPHA=0.2

srun apptainer exec --nv --cleanenv \
    -B $PROJECT:$PROJECT \
    $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
    /opt/conda/bin/python \
    $PROJECT/ijepa-ramp-stylegan2/train.py \
        --outdir=$PROJECT/outputs/Chest/sem_mixing_${SEM_MIX}/run_%j \
        --data=/scratch/<user>/dataset/256/chest_xray_labelled.zip \
        --gpus=4 --cond=1 \
        --ijepa_checkpoint /scratch/<user>/ijepa_backbone.pth \
        --ijepa_lambda 1.0 --ijepa_image 256 \
        --extra_dim 2048 --ijepa_warmup_kimg 5.4 \
        --sem_mixing_prob ${SEM_MIX} --fusion_alpha ${FUSION_ALPHA}
```

**Steps to launch**

1. Edit dataset paths / checkpoint paths.
2. Pick your `SEM_MIX` and `FUSION_ALPHA`.
3. Submit → `sbatch scripts/StyleGAN2/ijepa-ramp-stylegan2/your_job.sh`.

The script will:
* spin up 1 node × 4 GPUs  
* mount the project into the Apptainer container (`stylegan2ada-devel.sif`)  
* resume from `--resume` if provided, else start fresh  
* write snapshots / metrics in `outputs/…/training-runs/`  

*Experiments in the thesis ran on 4× A100 (32 GB); single‑GPU training is possible with `--batch_gpu 8`.*

### 4.3 Evaluation & Metrics (SLURM)

To compute Precision/Recall, KID, and IS on your trained checkpoints via SLURM + Apptainer, create a job script like below (`scripts/StyleGAN2/ijepa-ramp-stylegan2/metrics_job.sh`):

```bash
#!/bin/bash

# 1️⃣  Precision/Recall @50k (conditional)
srun --gres=gpu:2 \
  apptainer exec --nv --cleanenv \
    -B $PROJECT:$PROJECT \
    $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
    python /scratch/<user>~/calc_metrics_pr.py \
      --metrics=pr50k3_full_cond \
      --network=/scratch/<user>~/network-snapshot-000403.pkl \
      --data=/scratch/<user>~/dataset/256/Brain_cancer_labelled.zip

# 2️⃣  KID & IS @50k
srun --gres=gpu:2 \
  apptainer exec --nv --cleanenv \
    -B $PROJECT:$PROJECT \
    $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
    python /scratch/<user>~/calc_metrics.py \
      --metrics=kid50k_full,is50k \
      --network=/scratch/<user>~/network-snapshot-000403.pkl \
      --data=/scratch/<user>~/dataset/256/Brain_cancer_labelled.zip
```
---

## 5 Results snapshot 📊  

| Model (Chest X‑ray 256²) | FID ↓ | KID ×10³ ↓ | IS ↑ |
|--------------------------|-------|------------|------|
| StyleGAN2‑ADA baseline   | **5.63** | 3.2 | 2.32 |
| **Diff‑Proj‑FastGAN + JEPA** | 3.76 | **0.4** | 2.30 |
| Diff‑StyleGAN2 baseline  | 10.09 | 8.3 | 2.31 |

*(Complete tables and per‑class precision/recall in the thesis, Section 6.)*

---

## 6 Reproducing the thesis 📖  

Exact code, seeds `[2025, 425, 9001]`, and SLURM scripts used for every figure and table are archived **here** → <https://github.com/08Abhinay/IJEPA-Diffusion-GAN>.  
Run:

```bash
make all   # generates every figure & metric in ~18 h on 4 GPUs
```

---

## 7 Citation 📝  

```bibtex
@article{Belde2025,
author = "Abhinay Shankar Belde",
title = "{Addressing Data Scarcity in Medical Imaging: A Hybrid Approach Combining IJEPA, Diffusion, and GANs}",
year = "2025",
month = "7",
url = "https://hammer.purdue.edu/articles/thesis/Addressing_Data_Scarcity_in_Medical_Imaging_A_Hybrid_Approach_Combining_IJEPA_Diffusion_and_GANs/29649557",
doi = "10.25394/PGS.29649557.v1"
}
```


---


## 8 Acknowledgments 🙏  

Thanks to **Dr. Mohammadreza Hajiarbabi**, **Dr. Jonathan Rusert**, and **Dr. Alessandro Selvitella** for invaluable guidance, and to Purdue’s Gilbreth HPC staff for compute support.

This work builds upon several outstanding open-source repositories and their authors:
- **StyleGAN2-ADA** by Karras et al. (https://github.com/NVlabs/stylegan2-ada)  
- **InsGen** by Yang et al. (https://github.com/ceyuanyang/InsGen)  
- **ProjectedGAN** by Sauer et al. (https://github.com/axelsauer/projected-gan)  
- **Diffusion-GAN** by Wang et al. (https://github.com/Zhendong-Wang/Diffusion-GAN)

