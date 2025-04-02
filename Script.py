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


# Load the TSV file
excel_path = 'C:\\Users\\ludov\\Desktop\\OpenDataset\\participants.xlsx'  # double backslashes
# Read the Excel file
df = pd.read_excel(excel_path, usecols=['participant_id', 'group'])

# Mappa filename → label numerico (hc → 0, fcd → 1)
label_mapping = {row['participant_id']: 1 if row['group'] == 'fcd' else 0 for _, row in df.iterrows()}


# Definizione del Dataset
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(192, 256, 256)):
        """
        :param root_dir: Directory principale con le cartelle dei soggetti
        :param transform: Eventuali trasformazioni da applicare
        :param target_size: Dimensione target per il ridimensionamento (H, W, D)
        """
        self.root_dir = root_dir
        self.label_mapping = label_mapping  # Add label mapping
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
        img_name = os.path.basename(img_path)  # Extract filename
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

        # Get label from mapping
        label = self.label_mapping.get(img_name, 0)  # Default to 0 if not found

        # Ottieni la shape dell'immagine (utilizzare la shape del tensor PyTorch)
        #print("Shape dell'immagine:", img.shape)



        return img, torch.tensor(label, dtype=torch.long)


# Definizione delle trasformazioni
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Rotazione casuale ±15°
]) 


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
num_classes = 2
lr = 0.001
patience = 5
num_folds = 5

# Load dataset
root_dir = r'C:\Users\ludov\Desktop\OpenDataset'
dataset = NiftiDataset(root_dir=root_dir)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}/{num_folds}")
    
    train_size = int(len(train_idx) * 0.8)
    val_size = len(train_idx) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_idx, [train_size, val_size])

    train_loader = DataLoader(Subset(dataset, train_subset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_subset), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

   # Verifica caricamento dati
    for batch in train_loader:
        images, labels = batch  # Separare immagini e etichette
        print("Shape del batch:", images.shape)  # Dovrebbe essere [batch_size, 1, depth, height, width]
        break

    # Visualizzazione del primo slice 2D del primo esempio nel batch
    first_image = images[0, 0, :, :, :]  # Seleziona il primo esempio e il primo canale (in caso di immagini 1 canale)
    
    # Seleziona il primo slice in profondità (depth)
    first_slice = first_image[100, :, :]  # Prendi il primo slice lungo la profondità (profondità = 0)

    plt.figure()
    plt.imshow(first_slice, cmap='gray')
    plt.title("Slice at Depth 0")
    plt.axis('off')  # Disabilita gli assi
    plt.show()

    
    # Build model
    model = resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=2)
    print(model)
    summary(model, (1, 192, 256, 256))
    model.to(device)

    # Check GPU
    print(torch.cuda.is_available())  # Returns True if a GPU is detected
    print(torch.cuda.device_count())  # Number of available GPUs
    print(torch.cuda.get_device_name(0))  # Name of the first GPU
    print(torch.cuda.current_device())  # Index of the currently selected GPU

    # Caricamento dei pesi pre-addestrati
    checkpoint = torch.load("resnet_50_23dataset.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    # Reinizializzazione manuale di FC
    torch.nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    model.fc.bias.data.zero_()
    
    # Freezing dell'encoder (tutti i layer convoluzionali)
    for name, param in model.named_parameters():
        if not name.startswith("fc") and not name.startswith("avgpool"):
            param.requires_grad = False

    # Mostra il modello e i parametri aggiornabili
    summary(model, (1, 192, 256, 256))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    
    """ # Carica il checkpoint e prendi il 'state_dict'
    checkpoint = torch.load("resnet_50_23dataset.pth")

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # Carica i pesi nel modello ignorando i layer mancanti
    model.load_state_dict(checkpoint, strict=False)    """ 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    torch.cuda.synchronize()

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for idx, (images, labels) in enumerate(train_loader): 
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"Processing batch {idx+1}/{len(train_loader)}")
            
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = torch.randint(0, num_classes, (images.shape[0],)).to(device) ###
                outputs = model(images)
                loss = criterion(outputs, labels)
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


