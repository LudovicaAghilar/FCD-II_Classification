import os
import torch
import torchio as tio
import matplotlib.pyplot as plt
from medcam import medcam
from resnet import resnet18  # Assicurati che il tuo resnet.py abbia questa funzione

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

volume = subject['image'].data  # shape: (1, D, H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume = volume.to(device).unsqueeze(0)  # shape: (1, 1, D, H, W)

# === 3. Carica modello e pesi ===
model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
best_weights = torch.load(f"best_model_fold_{0}.pth", map_location=device)
model.load_state_dict(best_weights['state_dict'])
model = model.to(device)
model.eval()

# === 4. Crea cartella per salvare Grad-CAM ===
output_dir = r'C:\Users\ludov\Scripts\attention_maps'
os.makedirs(output_dir, exist_ok=True)

# === 5. Inietta Grad-CAM sul layer 'layer4' ===
model = medcam.inject(
    model,
    output_dir=output_dir,
    backend='gcam',
    layer='layer3',
    label='best',
    save_maps=True,
    return_attention=True
)

# === 6. Inferenza e ottenimento Grad-CAM ===
with torch.no_grad():
    output_logits, cam = model(volume)  # cam shape: (1, D, H, W)

# === 7. Visualizzazione slice centrale ===
cam_np = cam.squeeze().cpu().numpy()      # (D, H, W)
flair_np = volume.squeeze().cpu().numpy() # (D, H, W)

z = cam_np.shape[0] // 2  # slice centrale asse Z

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("FLAIR")
plt.imshow(flair_np[z], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM overlay")
plt.imshow(flair_np[z], cmap='gray')
plt.imshow(cam_np[z], cmap='hot', alpha=0.5)
plt.axis('off')

plt.tight_layout()
plt.show()

print("cam shape:", cam.shape)

plt.imshow(cam_np[z], cmap='hot')
plt.axis('off')
plt.show()

import torch.nn.functional as F

# Supponendo che `cam` sia di shape (1, D_small, H_small, W_small)
# e `volume` di shape (1, 1, D_orig, H_orig, W_orig)
cam_upsampled = F.interpolate(
    cam,  # (1, 1, D, H, W) per interpolate
    size=volume.shape[2:],  # (D, H, W)
    mode='trilinear',
    align_corners=False
).squeeze(1)  # torna a (1, D, H, W)

cam_image = tio.ScalarImage(tensor=cam_upsampled.cpu(), affine=subject['image'].affine)
cam_image.save('cam_resampled.nii.gz')
