import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# Definizione del Dataset
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(192, 256, 256)):
        """
        :param root_dir: Directory principale con le cartelle dei soggetti
        :param transform: Eventuali trasformazioni da applicare
        :param target_size: Dimensione target per il ridimensionamento (H, W, D)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.img_files = []

        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if os.path.isdir(subject_path):  # Controlla che sia una cartella
                anat_path = os.path.join(subject_path, "anat")
                if os.path.isdir(anat_path):  # Controlla che la cartella "anat" esista
                    for file in os.listdir(anat_path):
                        if "T1w.nii" in file:  # Seleziona solo i file T1w
                            self.img_files.append(os.path.join(anat_path, file))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Caricamento dell'immagine NIfTI
        img_path = self.img_files[idx]
        img = nib.load(img_path).get_fdata()  # Ottieni i dati come numpy array

        # Z-score normalization (zero mean, unit variance)
        mean = np.mean(img)
        std = np.std(img) + 1e-8  # Evita divisione per zero
        img = (img - mean) / std

        # Aggiunge il canale: (1, H, W, D)
        img = np.expand_dims(img, axis=0)

        # Converti in tensore PyTorch
        img = torch.tensor(img, dtype=torch.float32)

        # Ridimensionamento
        img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)

        # Data augmentation (rotazione casuale su 3 assi)
        if self.transform:
            img = self.transform(img)

        return img

# Definizione delle trasformazioni
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Rotazione casuale ±15°
])

# Parametri
root_dir = r'C:\Users\ludov\Desktop\OpenDataset'  # Cambia con il percorso corretto
batch_size = 1  #_

import matplotlib.pyplot as plt

# Creazione del Dataset e DataLoader
dataset = NiftiDataset(root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

""" # Verifica caricamento dati
for batch in dataloader:
    print(batch.shape)  # Dovrebbe essere [batch_size, 1, 192, 256, 256]
    break

# Visualizzazione del primo slice 2D del primo esempio nel batch
plt.figure()
plt.imshow(batch[0, 0, :, :, 0], cmap='gray')  # Slicing al secondo indice della profondità (es. slice 1)
plt.title("Slice at Depth 1")
plt.axis('off')  # Disabilita gli assi
plt.show() """

import torch
from resnet import resnet50  # Se hai il tuo script resnet.py con questa funzione

# Assuming the dimensions of your target images are (192, 256, 256)
sample_input_D = 192  # Depth
sample_input_H = 256  # Height
sample_input_W = 256  # Width
num_classes = 2   

# Inizializza il modello con i parametri necessari
model = resnet50(sample_input_D=sample_input_D, 
                 sample_input_H=sample_input_H, 
                 sample_input_W=sample_input_W, 
                 num_seg_classes=num_classes)


# Carica il checkpoint e prendi il 'state_dict'
checkpoint = torch.load("resnet_50_23dataset.pth")

# Se il checkpoint ha una chiave 'state_dict', estraila
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# Rimuovi il prefisso 'module.' dai nomi dei parametri se presente
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

# Carica i pesi nel modello ignorando i layer mancanti
model.load_state_dict(checkpoint, strict=False)

print(model)
