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
from monai.transforms import RandRotate


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

# Inizializza la trasformazione RandRotate con un range di ±15 gradi
rand_rotate = RandRotate(range_x=np.radians(15), range_y=np.radians(15), range_z=np.radians(15), prob=1.0, keep_size=True)

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
    

    def __getitem__(self, idx):
        # Caricamento dell'immagine NIfTI
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path)  # Extract filename

        # Estrai l'ID del partecipante dal nome del file
        participant_id = img_name.split('_')[0]  # ID prima dell'underscore (es. "sub-00001")

        img = nib.load(img_path).get_fdata()  # Ottieni i dati come numpy array
    
        # Aggiunge il canale: (1, H, W, D)
        img = np.expand_dims(img, axis=0)

        # Converti in tensore PyTorch
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

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
            
        # Stampa per verifica
        print(f"Participant ID: {participant_id} - Label: {label}")

        # Restituisci l'immagine, il label (aggiungendo la dimensione di batch) e l'ID del partecipante
        return img, torch.tensor(label, dtype=torch.long).unsqueeze(0), participant_id




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
num_epochs = 10
batch_size = 4
#num_classes = 2
lr = 0.001
patience = 5
num_folds = 5

# Load dataset
root_dir = r'C:\Users\ludov\Desktop\OpenDataset'
dataset = NiftiDataset(root_dir=root_dir, transform=rand_rotate)
#dataset = NiftiDataset(root_dir=root_dir)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracies = []  # Lista per memorizzare le accuracies di ogni fold

# Cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}/{num_folds}")

    # Verifica la distribuzione delle etichette nel training set
    """ train_labels = [dataset[i][1].item() for i in train_idx]
    val_labels = [dataset[i][1].item() for i in test_idx]
    
    print(f"Class distribution in training fold {fold+1}:")
    print(f"Training - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"Validation - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}") """
    
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
    model = resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
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

    state_dict = checkpoint["state_dict"]  # o semplicemente checkpoint se è direttamente lo state_dict

    # Rimuove "module." da ogni chiave
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # rimuove il prefisso
        new_state_dict[new_key] = v

    # Ora puoi caricare i pesi nel tuo modello
    model.load_state_dict(new_state_dict, strict=False)

    #model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Reinizializzazione manuale di FC
    """ torch.nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    model.fc.bias.data.zero_() """
    
    # Freezing dell'encoder (tutti i layer convoluzionali tranne layer4 e fc)
    for name, param in model.named_parameters():
        if not (name.startswith("fc") or name.startswith("layer4") or name.startswith("avgpool")):
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

    # All'inizio del ciclo per ogni fold, aggiungi le liste per memorizzare le perdite per ogni epoca
    train_losses = []  # Per memorizzare le perdite di addestramento
    val_losses = []    # Per memorizzare le perdite di validazione

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for idx, (images, labels, participant_id) in enumerate(train_loader): 

            #print(f"Participant ID: {participant_id[0]}, Label: {labels.item()}")

            images, labels = images.to(device), labels.to(device)
            
            # Plot the first slice of the image after pre-processing
            """ img_to_show = images[0].cpu().detach().numpy()  # Convert to numpy for plotting
            plt.figure(figsize=(6, 6))
            plt.imshow(img_to_show[0,100,:, :], cmap='gray')  # Visualizza il primo slice in 2D (assumendo che l'immagine sia [C, H, W, D])
            plt.title(f"Processed Image Slice - Participant {participant_id[0]}")
            plt.axis('off')
            plt.show()  """

            # Plot the histogram of image intensities (before or after preprocessing)
            img_to_show = images[0].cpu().detach().numpy()  # Convert to numpy for plotting

            # Flatten the image to 1D to plot the intensity histogram
            img_flattened = img_to_show.flatten()

            # Create histogram of pixel intensities
            """ plt.figure(figsize=(6, 6))
            plt.hist(img_flattened, bins=100, color='gray', alpha=0.7)
            plt.title(f"Histogram of Intensities - Participant {participant_id[0]}")
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.show() """
                
            optimizer.zero_grad()
            outputs = model(images)

            #outputs = model(images).squeeze()  # Remove the extra dimension, making it [batch_size]
            #print(f"Output shape: {outputs.shape}")  # Dovrebbe essere [batch_size, 1]


            #print(f"Participant ID: {participant_id[0]}, Output: {outputs}")  # Stampa anche l'ID

            #summary(model, (1, 192, 256, 256))
        
            #labels = labels.squeeze()  # Ensure the labels also have shape [batch_size]
            #print(f"Labels shape: {labels.shape}")  # Dovrebbe essere [batch_size, 1]


             # Calcola la perdita
            loss = criterion(outputs, labels.float())  # Squeeze per eliminare la dimensione 1, se necessario
            optimizer.zero_grad()
            loss.backward()

             # Print gradients of parameters after backward pass
            """ if idx % 10 == 0:  # Print gradients every 10 batches (or set any other condition)
                print(f"\nGradients after batch {idx+1}/{len(train_loader)}:")
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Ensure the parameter has a gradient
                        print(f"{name} - Gradient: {param.grad.abs().mean().item():.6f}")  # Print the mean absolute gradient for each parameter
 """
            optimizer.step()
            running_loss += loss.item()

             # Aggiungi il print per mostrare l'output e l'ID del partecipante
            print(f"[TRAIN] Participant ID: {participant_id[0]} - Output: {outputs.detach().cpu().numpy()} - Loss: {loss.item():.4f}")

            print(f"Processing batch {idx+1}/{len(train_loader)}")
            #  AGGIUNGI QUESTO BLOCCO PER STAMPARE OUTPUT, LOSS E ID
            #print(f"[TRAIN] Batch {idx+1}/{len(train_loader)} - Participant: {participant_id[0]}")
            #print(f"        Output: {outputs.detach().cpu().numpy()}")
            #print(f"        Label: {labels.item()} | Loss: {loss.item():.4f}")
            #print("-" * 60)
            
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Aggiungi la loss di addestramento alla lista

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for images, labels, participant_id in val_loader:

                images, labels = images.to(device), labels.to(device)
                #labels = torch.randint(0, num_classes, (images.shape[0],)).to(device) ###
                outputs = model(images)
                #outputs = model(images).squeeze()  # Remove the extra dimension, making it [batch_size]
                #labels = labels.squeeze()  # Ensure the labels also have shape [batch_size]
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)    
        val_losses.append(avg_val_loss)  # Aggiungi la loss di validazione alla lista
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

    # Al termine del training del fold, puoi tracciare le perdite per il fold corrente
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch - Fold {fold + 1}')
    plt.legend()
    plt.show()

    # Aggiungi la fase di testing dopo il training del fold
    # Aggiungi la fase di testing dopo il training del fold
    model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
    model.eval()
    correct = 0
    total = 0

    # Test phase per calcolare l'accuratezza
    with torch.no_grad():
        for batch_idx, (images, labels, participant_id) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # Applichiamo la funzione sigmoid per ottenere probabilità
            predicted = torch.sigmoid(outputs).round()  # Usa round per ottenere 0 o 1 (binary classification)
            batch_correct = (predicted.squeeze() == labels.squeeze()).sum().item()

            total += labels.size(0)  # Aggiungi il numero di esempi
            correct += (predicted == labels).sum().item()  # Conta i corretti

            print(f"[FOLD {fold}] Test Batch {batch_idx+1}/{len(test_loader)} - Participant: {participant_id[0]}")
            print(f"         Predicted: {predicted.squeeze().item()}, True: {labels.squeeze().item()}, Correct in Batch: {batch_correct}")

    accuracy = correct / total  # Calcola l'accuratezza per questo fold
    accuracies.append(accuracy)  # Aggiungi l'accuratezza di questo fold alla lista

    print(f"Fold {fold+1} Accuracy: {accuracy * 100:.2f}%")

# Calcola la media delle accuratezze
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy across all folds: {mean_accuracy * 100:.2f}%")