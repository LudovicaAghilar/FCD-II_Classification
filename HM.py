import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import animation
import torchio as tio
from resnet import resnet18   # Assicurati che resnet.py sia nel PYTHONPATH
from torchinfo import summary

# --------------------- PARAMETRI ---------------------
DATA_DIR       = r"C:\Users\ludov\Desktop\T1w_images"
WEIGHTS_PATH   = r"C:\Users\ludov\Scripts\Trial_T1w_Znorm\best_model_fold_1.pth"
TEST_TXT_PATH  = r"C:\Users\ludov\Scripts\Trial_T1w_Znorm\test_predictions_fold_1.txt"
OUT_DIR        = r"C:\Users\ludov\Desktop\T1w_gradcam_gif"
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
            subj_id = line.split(",")[0].strip()  # "sub-00003"
            ids.add(subj_id)
    return ids

TEST_IDS = load_test_ids(TEST_TXT_PATH)
print(f"Trovati {len(TEST_IDS)} ID nel file di test.")
# -----------------------------------------------------


# ---------------- CLASSE GRAD-CAM 3D -----------------
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(fwd_hook)
                module.register_full_backward_hook(bwd_hook)

    def generate(self, x):
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits.squeeze().backward(retain_graph=True)

        pooled = self.gradients.mean(dim=[0, 2, 3, 4])       # [C]
        heatmap = torch.einsum('cdhw,c->dhw', self.activations[0], pooled)
        heatmap = torch.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.cpu().numpy()
# -----------------------------------------------------


# ---------------- PREPROCESSING ----------------------
preproc = tio.Compose([
    tio.Resample((1.0, 1.0, 1.0)),
    tio.CropOrPad((160, 256, 256)),
    tio.ZNormalization(),
])

def load_volume(path):
    subject = tio.Subject(img=tio.ScalarImage(path))
    return preproc(subject)['img'].data       # (1, D, H, W)

def upsample_heatmap(hm, target_shape):
    hm = np.uint8(255 * hm)
    factors = tuple(t/s for t, s in zip(target_shape, hm.shape))
    return zoom(hm, factors, order=1)

def save_gif(vol_np, hm_np, out_path, alpha=0.4, fps=10, title="GradCAM"):
    fig, frames = plt.figure(figsize=(5, 5)), []
    for i in range(vol_np.shape[0]):
        fr = [
            plt.imshow(vol_np[i], cmap="gray", animated=True),
            plt.imshow(hm_np[i], cmap="jet", alpha=alpha, animated=True),
        ]
        frames.append(fr)
    plt.axis("off")
    plt.title(title)
    ani = animation.ArtistAnimation(fig, frames, interval=1000//fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
# -----------------------------------------------------


# ---------------- MODELLO & GRAD-CAM -----------------
model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256,
                 num_seg_classes=1).to(DEVICE)
# Example for a 3D input: batch_size=1, channels=1, depth=160, height=256, width=256
summary(model, input_size=(1, 1, 160, 256, 256))

state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state["state_dict"])
model.eval()

gradcam = GradCAM3D(model, TARGET_LAYER)
# -----------------------------------------------------


# ---------------- LOOP SUI VOLUMI --------------------
nii_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii*")))
print(f"Nella cartella immagini trovati {len(nii_files)} file.")

for path in nii_files:
    base = os.path.basename(path)
    subj_prefix = base.split("_")[0]          # "sub-00003"

    if subj_prefix not in TEST_IDS:
        print(f"{subj_prefix}: non è nel set di test → skip.")
        continue

    out_gif = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam.gif")
    if os.path.exists(out_gif):
        print(f"{subj_prefix}: GIF già esistente → skip.")
        continue

    try:
        print(f"{subj_prefix}: preprocessing & Grad‑CAM…")
        vol = load_volume(path).unsqueeze(0).to(DEVICE)   # (1,1,D,H,W)

        with torch.no_grad():
            _ = model(vol)                                # forward (class score)

        heatmap = gradcam.generate(vol)                   # (D,H,W)
        heatmap = upsample_heatmap(heatmap, vol.shape[-3:])  # resize to (160,256,256)

        vol_np = vol[0, 0].cpu().numpy()                  # (D,H,W)
        hm_np  = heatmap.astype(np.float32) / 255.0

        print(f"{subj_prefix}: salvo GIF…")
        save_gif(vol_np, hm_np, out_gif, alpha=ALPHA, fps=FPS,
                 title=f"{subj_prefix} Grad‑CAM")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore con {subj_prefix}: {e}")
# -----------------------------------------------------
