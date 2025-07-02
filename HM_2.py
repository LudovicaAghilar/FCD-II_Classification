# -*- coding: utf-8 -*-
"""
Genera heatmap Grad‑CAM 3D (in formato NIfTI) e GIF di overlay per tutte le
immagini di test elencate nel file `test_predicitions_fold_0`.

➤ Personalizza le sezioni PARAMS per adattare i percorsi al tuo ambiente.
   Lo script può essere lanciato da riga di comando o importato.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from math import ceil
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# ⬇⬇⬇ PARAMS – MODIFICA SOLO QUI ⬇⬇⬇
# ----------------------------------------------------------------------------
PREDICTIONS_FILE = Path(r"C:\Users\ludov\Scripts\Trial_T1w_Znorm\test_predictions_fold_0.txt")
IMG_DIR = Path(r"C:\Users\ludov\Desktop\T1w_images")  # cartella con le immagini *.nii.gz
OUTPUT_DIR = Path(r"C:\Users\ludov\Scripts\gradcam_outputs_GIF")
MODEL_WEIGHTS = Path(r"C:\Users\ludov\Scripts\Trial_T1w_Znorm\best_model_fold_0.pth")
BATCH_SIZE = 1                                           # numero soggetti in RAM
GPU = torch.cuda.is_available()
# ----------------------------------------------------------------------------
# ⬆⬆⬆ PARAMS – MODIFICA SOLO QUI ⬆⬆⬆
# ----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 🔧 MODELLO — carica e prepara rete con Grad‑CAM
# ---------------------------------------------------------------------------
from resnet import resnet18  # Assicurati che implementi i metodi Grad‑CAM

model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
weights = torch.load(str(MODEL_WEIGHTS), map_location="cpu")
model.load_state_dict(weights["state_dict"])
model.eval()
model = model.cuda() if GPU else model

# ---------------------------------------------------------------------------
# 🔧 TRANSFORMAZIONI PRE‑PROCESS
# ---------------------------------------------------------------------------
preprocess_transform = tio.Compose([
    tio.Resample((1.0, 1.0, 1.0)),
    tio.CropOrPad((160, 256, 256)),
    tio.ZNormalization(),
])

# ---------------------------------------------------------------------------
# 🔧 FUNZIONI DI UTILITÀ
# ---------------------------------------------------------------------------

def parse_predictions_file(path: Path) -> List[Tuple[str, int]]:
    """Estrae (subID, trueLabel) da ciascuna riga del file previsioni."""
    pattern = re.compile(r"^(sub-\d+),\s*True:\s*([01]\.0)")
    records: List[Tuple[str, int]] = []
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue  # ignora righe mal formattate
        sub_id, true_lbl = m.groups()
        records.append((sub_id, int(float(true_lbl))))
    return records


# ---------- Grad‑CAM base ----------

def generate_gradcam_heatmap(net: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """Restituisce heatmap 3D normalizzata [0‑1] come torch.Tensor (D×H×W)."""
    net.eval()
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    img = img.cuda(non_blocking=True) if GPU else img
    pred = net(x=img, reg_hook=True)
    pred[:, pred.argmax(dim=1)].backward()

    gradients = net.get_activations_gradient().cpu()
    activations = net.get_activations().cpu()
    pooled_gradients = gradients.mean(dim=(2, 3, 4), keepdim=True)
    weighted_activations = activations * pooled_gradients
    heatmap = F.relu(weighted_activations.mean(dim=1))

    # normalizza 0‑1
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-6
    return heatmap.squeeze(0)  # (D×H×W)


# ---------- GIF helper ----------

def save_gif(vol_np: np.ndarray, hm_np: np.ndarray, out_path: Path, alpha: float = 0.4, fps: int = 10, title: str = "GradCAM") -> None:
    """Crea GIF overlay salva su disco."""
    fig, frames = plt.figure(figsize=(5, 5)), []
    for i in range(vol_np.shape[0]):
        fr = [
            plt.imshow(vol_np[i], cmap="gray", animated=True),
            plt.imshow(hm_np[i], cmap="jet", alpha=alpha, animated=True),
        ]
        frames.append(fr)
    plt.axis("off")
    plt.title(title)
    ani = animation.ArtistAnimation(fig, frames, interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()


# ---------- Pipeline principale ----------

def main():
    subjects_info = parse_predictions_file(PREDICTIONS_FILE)
    if not subjects_info:
        raise RuntimeError(f"Nessun soggetto trovato in {PREDICTIONS_FILE}")

    # Crea dataset TorchIO
    subjects = []
    for sub_id, true_lbl in subjects_info:
        matching_files = list(IMG_DIR.glob(f"{sub_id}*.nii.gz"))
        if not matching_files:
            print(f"[WARN] Nessun file trovato per {sub_id} — skipping.")
            continue
        img_path = matching_files[0]  # usa il primo file trovato

        subjects.append(
            tio.Subject(img=tio.ScalarImage(str(img_path)), label=true_lbl, sub_id=sub_id)
        )

    loader = tio.SubjectsLoader(tio.SubjectsDataset(subjects, transform=preprocess_transform), batch_size=BATCH_SIZE, shuffle=False)

    spicejet_colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    spicejet_cmap = LinearSegmentedColormap.from_list("SpiceJet", spicejet_colors)

    for batch in loader:
        imgs = batch["img"]["data"]  # shape B×1×D×H×W
        subs = batch["sub_id"]  # lista ID soggetti
        heatmaps = generate_gradcam_heatmap(model, imgs)  # torna (B×D×H×W) ma B=1 se bs=1
        if heatmaps.ndim == 3:
            heatmaps = heatmaps.unsqueeze(0)

        for idx in range(heatmaps.shape[0]):
            sub = subs[idx]
            hm = heatmaps[idx].detach().cpu().numpy()
            vol = imgs[idx].squeeze().numpy()

            # Salva heatmap come NIfTI
            affine = batch["img"]["affine"][idx]
            hm_nifti_path = OUTPUT_DIR / f"gradcam_{sub}.nii.gz"
            nib.save(nib.Nifti1Image(hm, affine), str(hm_nifti_path))
            print(f"[✔] Heatmap salvata ➜ {hm_nifti_path}")

            # Crea overlay RGB per ispezione rapida (facoltativo)
            overlay_rgb = np.stack([(vol - vol.min()) / (vol.max() - vol.min())] * 3, axis=-1)

            import torch.nn.functional as F

            # Resize heatmap to match the input volume shape
            vol_shape = vol.shape  # (D, H, W)
            hm_torch = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0).float()  # shape: 1×1×D×H×W
            hm_resized = F.interpolate(hm_torch, size=vol_shape, mode="trilinear", align_corners=False)
            hm = hm_resized.squeeze().numpy()

            hm_norm = hm  # già 0‑1
            hm_color = spicejet_cmap(hm_norm)[..., :3]  # RGBA ➜ RGB
            overlay_rgb = np.clip(0.6 * overlay_rgb + 0.4 * hm_color, 0, 1)

            overlay_path = OUTPUT_DIR / f"overlay_{sub}.nii.gz"
            # --------------------------- con questo -----------------------------------------
            rgb_uint8 = (overlay_rgb * 255).astype(np.uint8)          # shape (D, H, W, 3)
            nib.save(nib.Nifti1Image(rgb_uint8, affine), str(overlay_path))

            # Salva GIF
            gif_path = OUTPUT_DIR / f"gradcam_{sub}.gif"
            save_gif(vol.astype(np.float32), hm, gif_path)
            print(f"[✔] GIF salvata ➜ {gif_path}\n")


if __name__ == "__main__":
    main()
