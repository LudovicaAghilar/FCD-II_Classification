import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import animation
import torchio as tio
import nibabel as nib
from resnet import resnet18   # Assicurati che resnet.py sia nel PYTHONPATH


# --------------------- PARAMETRI ---------------------
DATA_DIR       = r"C:\Users\ludov\Scripts\data_registered"
OUT_DIR        = r"C:\Users\ludov\Scripts\R1_crop_registration\HeatMap_avg"
TARGET_LAYER   = "layer4"           # Cambia se necessario
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA, FPS     = 0.4, 10
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------


# ---------------- LEGGI ID DI TEST -------------------
def load_test_ids(txt_path):
    ids = set()
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            subj_id = line.split(",")[0].strip()
            ids.add(subj_id)
    return ids

# Usa uno qualunque dei file test_predictions (contengono gli stessi ID)
TEST_TXT_PATH  = r"C:\Users\ludov\Scripts\R1_crop_registration\test_predictions_fold_0.txt"
TEST_IDS = load_test_ids(TEST_TXT_PATH)
print(f"Trovati {len(TEST_IDS)} ID nel file di test.")
# -----------------------------------------------------


# ---------------- CLASSE GRAD-CAM 3D -----------------
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.tlayer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.tlayer.register_forward_hook(fwd_hook))
        self.hooks.append(self.tlayer.register_full_backward_hook(bwd_hook))

    def generate(self, x, class_idx):
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        pooled = self.gradients.mean(dim=[0, 2, 3, 4])       # [C]
        heatmap = torch.einsum('cdhw,c->dhw', self.activations[0], pooled)
        heatmap = torch.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.cpu().numpy()
    
    def close(self):
        for h in self.hooks:
            h.remove()
# -----------------------------------------------------


# ---------------- PREPROCESSING ----------------------
preproc = tio.Compose([
    tio.ZNormalization(),
])

padding = tio.CropOrPad(
    (173,203,176),
    only_pad='True'
)

def load_volume(path):
    subject = tio.Subject(img=tio.ScalarImage(path))

    """ if "3TLE_NIGUARDA" in path or "3TLE_HC" in path:
        tensor = subject['img'].data
        affine = subject['img'].affine
        tensor = tensor.permute(0, 3, 1, 2)  # (1, x, y, z)
        subject['img'] = tio.ScalarImage(tensor=tensor, affine=affine)
        subject = padding(subject)
        print(f"Immagine trasformata: {path}: {subject['img'].shape}") """

    return preproc(subject)['img'].data  # (1, D, H, W)


def upsample_heatmap(hm, target_shape):
    #hm = np.uint8(255 * hm)
    factors = tuple(t/s for t, s in zip(target_shape, hm.shape))
    return zoom(hm, factors, order=1)

import math

def show_all_slices_overlay(volume, heatmap, alpha=0.4, cmap_heat="jet"):
    """
    Visualizza tutte le slice di un volume 3D con overlay della heatmap.
    volume : numpy array con shape (D, H, W) -> immagine originale
    heatmap: numpy array con shape (D, H, W) -> heatmap normalizzata (0-1 o 0-255)
    """
    import math
    D = volume.shape[0]
    cols = math.ceil(math.sqrt(D))
    rows = math.ceil(D / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    # normalizza volume su [0,1] per la visualizzazione isn grigio
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    for i in range(D):
        axes[i].imshow(vol_norm[i, :, :], cmap="gray")
        axes[i].imshow(heatmap[i, :, :], cmap=cmap_heat, alpha=alpha)
        axes[i].axis("off")
        axes[i].set_title(f"Slice {i}", fontsize=6)

    """ # nascondi subplot vuoti
    for j in range(D, len(axes)):
        axes[j].axis("off") """

    plt.tight_layout()
    plt.show()



# -----------------------------------------------------


# ---------------- CARICO I 5 MODELLI -----------------
FOLDS = range(5)
MODELS = []

for f in FOLDS:
    wpath = fr"C:\Users\ludov\Scripts\R1_crop_registration\best_model_fold_{f}.pth"
    state = torch.load(wpath, map_location=DEVICE)
    model = resnet18(sample_input_D=173, sample_input_H=203, sample_input_W=176,
                     num_seg_classes=1).to(DEVICE)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    MODELS.append(GradCAM3D(model, model.layer4))
print(f"Caricati {len(MODELS)} modelli (fold 0–4).")
# -----------------------------------------------------


# ---------------- LOOP SUI VOLUMI --------------------
nii_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii*")))
print(f"Nella cartella immagini trovati {len(nii_files)} file.")

for path in nii_files:
    base = os.path.basename(path)
    subj_prefix = base.split(".")[0]

    if subj_prefix not in TEST_IDS:
        print(f"{subj_prefix}: non è nel set di test → skip.")
        continue 

    hm_raw_path = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam_mean.nii.gz")
    out_gif = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam_mean.gif")
    if os.path.exists(hm_raw_path):
        print(f"{subj_prefix}: Heatmap media già esistente → skip.")
        continue 

    try:
        print(f"{subj_prefix}: preprocessing & Grad-CAM media…")
        vol = load_volume(path).unsqueeze(0).to(DEVICE)   # (1,1,D,H,W)

        heatmaps = []
        for gcam in MODELS:
            hm = gcam.generate(vol, class_idx=0)

            """ if "3TLE_NIGUARDA" in path or "3TLE_HC" in path:
                hm = hm.transpose(1, 2, 0) """


            heatmaps.append(hm)

        # Media sulle fold
        heatmap_mean = np.mean(heatmaps, axis=0)

        # Salva la heatmap media RAW (prima dell'upsample)
        hm_raw_native_path = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam_mean_native.nii.gz")
        nib.save(
            nib.Nifti1Image(heatmap_mean.astype(np.float32), np.eye(4)), 
            hm_raw_native_path
        )

        # Upsample al volume originale
        img_nii = nib.load(path)
        heatmap_mean = upsample_heatmap(heatmap_mean, img_nii.shape)

        # Converte il volume in numpy per overlay
        vol_np = vol.cpu().numpy()[0, 0]   # (D,H,W)

        # Mostra overlay
        #show_all_slices_overlay(vol_np, heatmap_mean, alpha=0.4, cmap_heat="jet")


        # Salva la heatmap media
        nib.save(nib.Nifti1Image(heatmap_mean.astype(np.float32), img_nii.affine), hm_raw_path)

        """ # Salva anche GIF
        vol_np = vol.cpu().numpy()[0,0]
        save_gif(vol_np, heatmap_mean, out_gif, alpha=ALPHA, fps=FPS,
                 title=f"{subj_prefix} GradCAM Mean")
 """
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore con {subj_prefix}: {e}")




# -----------------------------------------------------

# Chiudo hook
for gcam in MODELS:
    gcam.close()
