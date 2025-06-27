import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.patches import Rectangle
import cv2
from resnet import resnet18  # Assicurati che il tuo resnet.py abbia questa funzione
import torchio as tio
import os

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[:, class_idx]
        score.backward()

        # Global average pooling dei gradienti su D, H, W
        pooled_grads = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)

        # Pesatura delle attivazioni
        weighted_activations = self.activations * pooled_grads
        heatmap = torch.sum(weighted_activations, dim=1).squeeze()

        # ReLU e normalizzazione
        heatmap = F.relu(heatmap)
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)

        return heatmap.cpu().numpy()

def get_resized_heatmap(heatmap, shape):
    """Resize heatmap to shape"""
    # Rescale heatmap to a range 0-255
    upscaled_heatmap = np.uint8(255 * heatmap)

    upscaled_heatmap = zoom(
        upscaled_heatmap,
        (
            shape[0] / upscaled_heatmap.shape[0],
            shape[1] / upscaled_heatmap.shape[1],
            shape[2] / upscaled_heatmap.shape[2],
        ),
    )

    return upscaled_heatmap


img_path = r"C:\Users\ludov\OneDrive\Desktop\CNR\sub-00014_acq-sag111_T1w.nii.gz"

# === 2. Preprocessing ===
subject = tio.Subject(image=tio.ScalarImage(img_path))
transform = tio.Compose([
    tio.Resample((1.0, 1.0, 1.0),),
    tio.CropOrPad((160, 256, 256)),
    #tio.ZNormalization()
])
subject = transform(subject)

volume = subject['image'].data  # shape: (1, D, H, W)
plt.figure()
plt.imshow(volume[0, 80, :, :], cmap='gray')  # Visualizza slice centrale
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume = volume.to(device).unsqueeze(0)  # shape: (1, 1, D, H, W)

model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
path = r"C:\Users\ludov\OneDrive\Desktop\CNR\best_model_fold_0.pth"
best_weights = torch.load(path, map_location=device)
model.load_state_dict(best_weights['state_dict'])
model = model.to(device)
model.eval()
output = model(volume)
print(output)
# Supponiamo che l'ultimo layer convoluzionale si chiami 'layer4'

target_layer = 'layer4'  # Modifica se necessario in base alla tua ResNet 3D
gradcam = GradCAM3D(model, target_layer)

# Inference e Grad-CAM
output = model(volume)
probs = torch.sigmoid(output)
predicted = probs.round()
print("Predicted:", predicted)

# Generate Grad-CAM heatmap
heatmap = gradcam.generate(volume)

heatmap_res = get_resized_heatmap(heatmap,(160,256,256))


slice_idx = 102  # Slice index for visualization

# Plot overlay
plt.figure()
plt.imshow(volume[0,0, slice_idx, :, :], cmap='gray')  # Base anatomical image
plt.imshow(heatmap_res[slice_idx, :, :], cmap='jet', alpha=0.5)  # Heatmap overlay
plt.title(f'Overlay at slice {slice_idx}')
plt.axis('off')
plt.show()

import nibabel as nib

# Recupera l'affine originale
affine = subject['image'].affine

# Crea immagine NIfTI
nifti_img = nib.Nifti1Image(heatmap.astype(np.float32), affine)

# Salva su disco
output_path = r"C:\Users\ludov\OneDrive\Desktop\CNR\gradcam_heatmap.nii.gz"
nib.save(nifti_img, output_path)

print(f"Heatmap salvata in: {output_path}")