# IJEPA-Diffusion-GAN 🩺✨  
*A hybrid generative framework for anatomy‑aware, data‑efficient medical‑image synthesis*

![License](https://img.shields.io/github/license/08Abhinay/IJEPA-Diffusion-GAN)
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

## 4 Quick‑start 🔧  

```bash
# 1⃣  Create env (PyTorch 2.2 + CUDA 12)
conda create -n ijepa_diffusion_gan python=3.10
conda activate ijepa_diffusion_gan
pip install -r requirements.txt

# 2⃣  Download pretrained I‑JEPA encoder (~120 MB)
bash scripts/download_ijepa.sh

# 3⃣  Train baseline StyleGAN2‑ADA + JEPA‑cosine on Chest X‑ray
python train.py --cfg configs/cxr_sg2_jepa.yaml

# 4⃣  Evaluate & log metrics
python metrics/eval_all.py --run_id <RUN_DIR>
```

*Experiments in the thesis ran on 4× V100 (32 GB); single‑GPU training is possible with `--batch_gpu 8`.*

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
@mastersthesis{belde2025ijepa,
  title  = {Addressing Data Scarcity in Medical Imaging: A Hybrid Approach Combining IJEPA, Diffusion, and GANs},
  author = {Abhinay Shankar Belde},
  school = {Purdue University},
  year   = {2025},
}
```

---

## 8 License 📄  

Apache 2.0 — see `LICENSE`.  
**Generated images must not be used for clinical diagnosis; they are research artifacts only.**

---

## 9 Acknowledgments 🙏  

Thanks to **Dr. Mohammadreza Hajiarbabi**, **Dr. Jonathan Rusert**, and **Dr. Alessandro Selvitella** for invaluable guidance, and to Purdue’s Gilbreth HPC staff for compute support.
