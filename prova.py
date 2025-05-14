import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import torchio as tio
from resnet import resnet18  # Assicurati che il tuo resnet.py abbia questa funzione

# === 1. Caricamento immagine ===
root_dir = r"C:\Users\ludov\Desktop\flair_images"
crop_size = (160, 256, 256)

all_files = [f for f in os.listdir(root_dir) if f.endswith('_FLAIR.nii.gz')]
if not all_files:
    raise FileNotFoundError("Nessuna immagine FLAIR trovata nella cartella.")
img_path = os.path.join(root_dir, all_files[86])  # ⚠️ assicurati che all_files[100] esista

# === 2. Preprocessing ===
subject = tio.Subject(image=tio.ScalarImage(img_path))
transform = tio.Compose([
    tio.ZNormalization(),
    tio.CropOrPad(crop_size)
])
subject = transform(subject)

volume = subject['image'].data  # shape: (1, D, H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume = volume.to(device).unsqueeze(0)  # shape: (1, 1, D, H, W)

# === 3. Carica modello e pesi ===
model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
best_weights = torch.load(f"best_model_fold_{0}.pth", map_location=device)
model.load_state_dict(best_weights['state_dict'])
model = model.to(device)
model.eval()

# === 1. Scegliamo un layer convoluzionale da cui prendere le feature maps ===
target_layer = model.layer3  # ad esempio l'ultimo blocco di ResNet18

# === 2. Hook per catturare attivazioni e gradienti ===
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Registrazione hook
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# === 3. Forward pass ===
output = model(volume)  # output: (1, 1)
class_idx = (output > 0).float()  # classificazione binaria
model.zero_grad()
output.backward(gradient=torch.ones_like(output))  # backward pass

# === 4. Calcolo Grad-CAM ===
grads_val = gradients[0]        # shape: (1, C, D', H', W')
activations_val = activations[0]  # shape: (1, C, D', H', W')

weights = grads_val.mean(dim=[2, 3, 4], keepdim=True)  # media spaziale
cam = (weights * activations_val).sum(dim=1, keepdim=True)  # shape: (1, 1, D', H', W')
cam = F.relu(cam)

# Normalizzazione e resize
# Normalizzazione e resize
cam = F.interpolate(cam, size=volume.shape[2:], mode='trilinear', align_corners=False)
cam = cam.squeeze().detach().cpu().numpy()  # ✅ fix qui
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)


# === 5. Visualizzazione slice-by-slice ===
volume_np = volume.squeeze().cpu().numpy()  # (D, H, W)

def plot_overlay(cam, volume, axis=0, num_slices=10):
    indices = np.linspace(0, volume.shape[axis]-1, num_slices, dtype=int)
    fig, axs = plt.subplots(1, num_slices, figsize=(15, 5))
    for i, idx in enumerate(indices):
        if axis == 0:
            img = volume[idx, :, :]
            heat = cam[idx, :, :]
        elif axis == 1:
            img = volume[:, idx, :]
            heat = cam[:, idx, :]
        elif axis == 2:
            img = volume[:, :, idx]
            heat = cam[:, :, idx]
        
        axs[i].imshow(img, cmap='gray')
        axs[i].imshow(heat, cmap='jet', alpha=0.5)
        axs[i].axis('off')
        axs[i].set_title(f"Slice {idx}")
    plt.tight_layout()
    plt.show()

# Esegui overlay su asse assiale (D)
plot_overlay(cam, volume_np, axis=0)
