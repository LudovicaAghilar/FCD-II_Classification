import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
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

# === 4. Implementazione Grad-CAM 3D ===
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if output is not None:
                self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)


    def generate_cam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)  # forward pass
        loss = output.squeeze()  # se preferisci rimuovere dimensioni inutili
        loss.backward()  # backward pass

        # Gradients shape: [1, C, D, H, W]
        weights = self.gradients.mean(dim=[2,3,4], keepdim=True)  # global average pooling over D,H,W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # weighted sum over channels
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalizza cam a [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# === 5. Crea istanza GradCAM e genera mappa ===
target_layer = model.layer3  # modifica se il nome è diverso

gradcam = GradCAM3D(model, target_layer)
cam = gradcam.generate_cam(volume)  # output shape (D_cam, H_cam, W_cam)

# === 6. Interpolazione GradCAM alla dimensione originale ===
cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,D_cam,H_cam,W_cam)
cam_resized = F.interpolate(cam_tensor, size=volume.shape[2:], mode='trilinear', align_corners=False)
cam_resized = cam_resized.squeeze().cpu().numpy()  # shape (D_orig, H_orig, W_orig)

# === 7. Visualizzazione ===
def show_gradcam_on_slice(volume, cam, alpha=0.4):
    """
    volume: numpy array (D, H, W) - immagine originale normalizzata
    cam: numpy array (D, H, W) - mappa GradCAM normalizzata [0,1]
    slice_idx: int - indice della slice da visualizzare (asse assiale)
    alpha: float - trasparenza della heatmap
    """
    """ if slice_idx is None:
        slice_idx = volume.shape[0] // 2  # slice centrale """

    img_slice = volume[:,170,:]
    cam_slice = cam[:,170,:]

    plt.figure(figsize=(6, 8))
    plt.imshow(img_slice, cmap='gray', interpolation='none')
    plt.imshow(cam_slice, cmap='jet', alpha=alpha, interpolation='none')
    plt.colorbar(label='Activation intensity')
    plt.axis('off')
    plt.show()

# Converti volume torch tensor a numpy per visualizzazione
volume_np = subject['image'].data.squeeze().cpu().numpy()

# Visualizza la slice centrale con GradCAM interpolata
show_gradcam_on_slice(volume_np, cam_resized)
