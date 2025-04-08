import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from resnet import resnet50  # Se hai il tuo script resnet.py con questa funzione
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import scipy.ndimage
import random
from collections import OrderedDict


def set_seed(seed):
    # Fissare il seed per la libreria standard Python
    random.seed(seed)
    
    # Fissare il seed per NumPy
    np.random.seed(seed)
    
    # Fissare il seed per PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per la modalità multi-GPU
    torch.backends.cudnn.deterministic = True  # Rendere deterministico per operazioni CUDA
    torch.backends.cudnn.benchmark = False  # Disabilitare ottimizzazioni di performance non deterministiche

# Esegui il seeding con un numero fisso (ad esempio 42)
set_seed(42)

# Load the TSV file
excel_path = 'C:\\Users\\ludov\\Desktop\\OpenDataset\\participants.xlsx'  # double backslashes
# Read the Excel file
df = pd.read_excel(excel_path, usecols=['participant_id', 'group'])

# Mappa filename → label numerico (hc → 0, fcd → 1)
label_mapping = {row['participant_id']: 1 if row['group'] == 'fcd' else 0 for _, row in df.iterrows()}


# Definizione del Dataset
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(192, 256, 256), rotation_range=15):
        """
        :param root_dir: Directory principale con le cartelle dei soggetti
        :param transform: Eventuali trasformazioni da applicare
        :param target_size: Dimensione target per il ridimensionamento (H, W, D)
        """
        self.root_dir = root_dir
        self.label_mapping = label_mapping  # Add label mapping
        self.transform = transform
        self.target_size = target_size
        self.rotation_range = rotation_range
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
    
    def random_rotation(self, img):
        """
        Applica una rotazione 3D casuale all'immagine.
        :param img: L'immagine NIfTI in formato numpy
        :return: Immagine ruotata
        """
        # Angoli di rotazione casuali per ciascun asse (in gradi)
        angle_x = random.uniform(-self.rotation_range, self.rotation_range)
        angle_y = random.uniform(-self.rotation_range, self.rotation_range)
        angle_z = random.uniform(-self.rotation_range, self.rotation_range)

        # Ruota l'immagine intorno agli assi X, Y, Z
        img = scipy.ndimage.rotate(img, angle_x, axes=(1, 2), reshape=False)  # Rotazione sull'asse X
        img = scipy.ndimage.rotate(img, angle_y, axes=(0, 2), reshape=False)  # Rotazione sull'asse Y
        img = scipy.ndimage.rotate(img, angle_z, axes=(0, 1), reshape=False)  # Rotazione sull'asse Z
        
        return img

    def __getitem__(self, idx):
        # Caricamento dell'immagine NIfTI
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path)  # Extract filename

        # Estrai l'ID del partecipante dal nome del file
        participant_id = img_name.split('_')[0]  # ID prima dell'underscore (es. "sub-00001")

        img = nib.load(img_path).get_fdata()  # Ottieni i dati come numpy array
    
        # Aggiunge il canale: (1, H, W, D)
        img = np.expand_dims(img, axis=0)

        # Applicare rotazione 3D casuale
        #img = self.random_rotation(img)

        # Converti in tensore PyTorch
        img = torch.tensor(img, dtype=torch.float32)

        # Ridimensionamento
        img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)

        # Z-score normalization (zero mean, unit variance) on PyTorch tensor
        mean = img.mean()
        std = img.std() + 1e-8  # Evita divisione per zero
        img = (img - mean) / std

        # Get label from mapping using participant_id
        label = self.label_mapping.get(participant_id, 0)  # Default to 0 if not found

        # Ottieni la shape dell'immagine (utilizzare la shape del tensor PyTorch)
        #print("Shape dell'immagine:", img.shape)

        return img, torch.tensor(label, dtype=torch.long), participant_id



""" # Verifica caricamento dati
for batch in dataloader:
    print(batch.shape)  # Dovrebbe essere [batch_size, 1, 192, 256, 256]
    break

# Visualizzazione del primo slice 2D del primo esempio nel batch
plt.figure()
plt.imshow(batch[0, 0, :, :, 0], cmap='gray')  # Slicing al secondo indice della profondità (es. slice 1)
plt.title("Slice at Depth 1")
plt.axis('off')  # Disabilita gli assi
plt.show() 
 """


# Training parameters
num_epochs = 50
batch_size = 1
#num_classes = 2
lr = 0.001
patience = 5
num_folds = 5

# Load dataset
root_dir = r'C:\Users\ludov\Desktop\OpenDataset'
dataset = NiftiDataset(root_dir=root_dir)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


# Cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}/{num_folds}")
    
    # Creazione dei sottoinsiemi per training, validazione e test
    test_subset = Subset(dataset, test_idx)
    train_val_subset = Subset(dataset, train_idx)
    
    # Suddivisione training-validation
    train_size = int(0.8 * len(train_val_subset))
    val_size = len(train_val_subset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_size, val_size])
    
    # Creazione dei DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    
    # Build model
    backbone = resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
    print(backbone)
    backbone.to(device)

    # Check GPU
    print(torch.cuda.is_available())  # Returns True if a GPU is detected
    print(torch.cuda.device_count())  # Number of available GPUs
    print(torch.cuda.get_device_name(0))  # Name of the first GPU
    print(torch.cuda.current_device())  # Index of the currently selected GPU

    # Carica i pesi
    checkpoint = torch.load("resnet_50_23dataset.pth", map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc")}  # Escludi fc
    backbone.load_state_dict(state_dict, strict=False)

    
    # Costruisci la testa classificatrice
    classifier_head = ClassifierHead(in_features=2048, num_classes=1).to(device)

    # Wrapper per unire tutto
    class FullModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            out = self.head(features)
            return out

    # Unisci backbone + fc
    model = FullModel(backbone, classifier_head).to(device)

    # Reinizializzazione manuale del layer FC
    torch.nn.init.kaiming_normal_(model.head.fc.weight, mode='fan_out', nonlinearity='relu')
    model.head.fc.bias.data.zero_()

    
    # Optional: Freezing
    for name, param in model.backbone.named_parameters():
        if not name.startswith("layer4") and not name.startswith("avgpool"):
            param.requires_grad = False


    # Mostra il modello e i parametri aggiornabili
    #summary(model, (1, 192, 256, 256))
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Trainable Parameters: {trainable_params}")

    
    """ # Carica il checkpoint e prendi il 'state_dict'
    checkpoint = torch.load("resnet_50_23dataset.pth")

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # Carica i pesi nel modello ignorando i layer mancanti
    model.load_state_dict(checkpoint, strict=False)    """ 
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    torch.cuda.synchronize()

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for idx, (images, labels, participant_id) in enumerate(train_loader): 

            #print(f"Participant ID: {participant_id[0]}, Label: {labels.item()}")

            images, labels = images.to(device), labels.to(device)
            
             # Plot the first slice of the image after pre-processing
            #img_to_show = images[0].cpu().detach().numpy()  # Convert to numpy for plotting
            #plt.figure(figsize=(6, 6))
            #plt.imshow(img_to_show[0,100,:, :], cmap='gray')  # Visualizza il primo slice in 2D (assumendo che l'immagine sia [C, H, W, D])
            #plt.title(f"Processed Image Slice - Participant {participant_id[0]}")
            #plt.axis('off')
            #plt.show()
            
            optimizer.zero_grad()
            outputs = model(images)

            print(outputs)

            outputs = model(images).squeeze()  # Remove the extra dimension, making it [batch_size]

            #print(f"Participant ID: {participant_id[0]}, Output: {outputs}")  # Stampa anche l'ID

            #summary(model, (1, 192, 256, 256))
        
            labels = labels.squeeze()  # Ensure the labels also have shape [batch_size]

             # Calcola la perdita
            loss = criterion(outputs, labels.float())  # Squeeze per eliminare la dimensione 1, se necessario
            optimizer.zero_grad()
            loss.backward()

             # Print gradients of parameters after backward pass
            if idx % 10 == 0:  # Print gradients every 10 batches (or set any other condition)
                print(f"\nGradients after batch {idx+1}/{len(train_loader)}:")
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Ensure the parameter has a gradient
                        print(f"{name} - Gradient: {param.grad.abs().mean().item():.6f}")  # Print the mean absolute gradient for each parameter

            optimizer.step()
            running_loss += loss.item()


            print(f"Processing batch {idx+1}/{len(train_loader)}")
            #  AGGIUNGI QUESTO BLOCCO PER STAMPARE OUTPUT, LOSS E ID
            #print(f"[TRAIN] Batch {idx+1}/{len(train_loader)} - Participant: {participant_id[0]}")
            #print(f"        Output: {outputs.detach().cpu().numpy()}")
            #print(f"        Label: {labels.item()} | Loss: {loss.item():.4f}")
            #print("-" * 60)
            
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for images, labels, participant_id in val_loader:

                images, labels = images.to(device), labels.to(device)
                #labels = torch.randint(0, num_classes, (images.shape[0],)).to(device) ###
                outputs = model(images)
                outputs = model(images).squeeze()  # Remove the extra dimension, making it [batch_size]
                labels = labels.squeeze()  # Ensure the labels also have shape [batch_size]
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break





""" if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

# Carica i pesi nel modello ignorando i layer mancanti
model.load_state_dict(checkpoint, strict=False)

print(model) """

 
