import torch
import torch.nn.functional as F
from resnet import resnet18  # Assicurati che il tuo script resnet.py contenga questa funzione
import torchio as tio
import os

import torch
import torch.nn.functional as F

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook forward
        def forward_hook(module, input, output):
            self.activations = output

        # Hook backward
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            # auto-infer classe predetta (assume classificazione)
            if output.shape[1] == 1:
                target = output[0, 0]
            else:
                target_class = output.argmax(dim=1).item()
                target = output[0, target_class]
        else:
            target = output[0, target_class]

        target.backward()

        # Check dimensioni attivazioni
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hook non ha catturato gradienti o attivazioni. Assicurati che il target_layer sia corretto.")

        # Calcolo CAM
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (B, 1, D, H, W)

        # Normalizza
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        # Resize al volume originale
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)

        return cam

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

# === 1. Trova la prima immagine FLAIR ===
root_dir = r"C:\Users\ludov\Desktop\flair_images"
crop_size = (160, 256, 256)

all_files = [f for f in os.listdir(root_dir) if f.endswith('_FLAIR.nii.gz')]
if not all_files:
    raise FileNotFoundError("Nessuna immagine FLAIR trovata nella cartella.")
img_path = os.path.join(root_dir, all_files[100])

# === 2. Preprocessing ===
subject = tio.Subject(image=tio.ScalarImage(img_path))
transform = tio.Compose([
    tio.ZNormalization(),
    tio.CropOrPad(crop_size)

])
subject = transform(subject)

volume = subject['image'].data  # (1, D, H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume = volume.to(device).unsqueeze(0)  # (1, 1, D, H, W)

# === 3. Carica e modifica il modello ===
model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
best_weights = torch.load(f"best_model_fold_{0}.pth")
best_names = best_weights['state_dict'] 
model.load_state_dict(best_names)
model = model.to(device)
model.eval()

# Esempio: layer target
target_layer = model.layer3[-1]  # non l'intero blocco!

gradcam = GradCAM3D(model, target_layer)
cam_volume = gradcam.generate_cam(volume)  # (1, 1, D, H, W)

gradcam.remove_hooks()  # evita memory leak

import matplotlib.pyplot as plt
import numpy as np

# Rimuovi batch e channel dimensioni
cam_np = cam_volume.squeeze().detach().cpu().numpy()
flair_np = volume.squeeze().cpu().numpy() # (D, H, W)

# Prendi lo slice centrale dell'asse Z
z = flair_np.shape[0] // 2

plt.figure(figsize=(10, 4))

# FLAIR puro
plt.subplot(1, 2, 1)
plt.title("FLAIR")
plt.imshow(flair_np[z+6], cmap='gray')

# CAM sovrapposto
plt.subplot(1, 2, 2)
plt.title("CAM overlay")
plt.imshow(flair_np[z+6], cmap='gray')
plt.imshow(cam_np[z+6], cmap='hot', alpha=0.5)

plt.tight_layout()
plt.show()
