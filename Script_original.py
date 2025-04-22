import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from resnet import resnet18# Se hai il tuo script resnet.py con questa funzione
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import scipy.ndimage
import random
from collections import OrderedDict
from monai.transforms import RandRotate
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold



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
#set_seed(42)


# Load the TSV file
excel_path = 'C:\\Users\\ludov\\Desktop\\OpenDataset\\participants.xlsx'  # double backslashes
# Read the Excel file
df = pd.read_excel(excel_path, usecols=['participant_id', 'group'])

# Mappa filename → label numerico (hc → 0, fcd → 1)
label_mapping = {row['participant_id']: 1 if row['group'] == 'fcd' else 0 for _, row in df.iterrows()}


# Definizione del Dataset
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory principale con le cartelle dei soggetti
        :param transform: Eventuali trasformazioni da applicare
        :param target_size: Dimensione target per il ridimensionamento (H, W, D)
        """
        self.root_dir = root_dir
        self.label_mapping = label_mapping  # Add label mapping
        self.img_files = []
        self.transform = transform

 
        # Cerca tutte le immagini che terminano con T1w_brain.nii.gz nella directory specificata
        for file in os.listdir(root_dir):
            if file.endswith("brain_preprocessed.nii.gz"):
                self.img_files.append(os.path.join(root_dir, file))


    def __len__(self):
        return len(self.img_files)
    

    def __getitem__(self, idx):
        # Caricamento dell'immagine NIfTI
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path)  # Extract filename

        # Estrai l'ID del partecipante dal nome del file
        participant_id = img_name.split('_')[0]  # ID prima dell'underscore (es. "sub-00001")
    
        # Carica l'immagine NIfTI usando nibabel
        img = nib.load(img_path).get_fdata()  # Carica l'immagine come array NumPy

        # Aggiunge il canale: (1, H, W, D)
        img = np.expand_dims(img, axis=0)

        # Converti in tensore PyTorch
        img = torch.tensor(img, dtype=torch.float32)

        # Applica la trasformazione se specificata
        if self.transform:
            img = self.transform(img)

        # Ridimensionamento
        #img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)

        # Get label from mapping using participant_id
        label = self.label_mapping.get(participant_id, 0)  # Default to 0 if not found
            
        # Stampa per verifica
        #print(f"Participant ID: {participant_id} - Label: {label}")

        # Restituisci l'immagine, il label (aggiungendo la dimensione di batch) e l'ID del partecipante
        return img, torch.tensor(label, dtype=torch.long).unsqueeze(0), participant_id


# Training parameters
num_epochs = 20
batch_size = 8
lr = 0.001
patience = 5
num_folds = 5

verbose=0

# Load dataset
root_dir = r"C:\Users\ludov\Desktop\Dataset_modified\preprocessed"
rand_rotate = RandRotate(range_x=np.radians(15), range_y=np.radians(15), range_z=np.radians(15), prob=1.0, keep_size=True)
dataset = NiftiDataset(root_dir=root_dir)
# Ottieni i label per ogni immagine in ordine di dataset.img_files
all_labels = [label_mapping[os.path.basename(path).split('_')[0]] for path in dataset.img_files]
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

def train_model(num_epochs, batch_size, lr, patience, optimizer, experiment_name='default_exp'):

    # Usa la GPU se presente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lista per memorizzare le accuracies di ogni fold
    accuracies = []  

    # Cross validatio
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, all_labels)):
        print(f"Training fold {fold+1}/{num_folds}")
        
        test_dataset = NiftiDataset(root_dir=root_dir, transform=None)
        train_dataset = NiftiDataset(root_dir=root_dir, transform=rand_rotate)
        val_dataset = NiftiDataset(root_dir=root_dir, transform=None)

        # Label relativi solo al training+val set
        train_val_labels = [all_labels[i] for i in train_idx]

        # Stratified split per ottenere train/val da train_val (80/20)
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(inner_skf.split(train_idx, train_val_labels))

        # Mappa agli indici originali del dataset
        train_indices = [train_idx[i] for i in inner_train_idx]
        val_indices = [train_idx[i] for i in inner_val_idx]


        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_idx)

        # Conta le classi nel train set
        label_counts = {0: 0, 1: 0}
        for i in train_indices:
            _, label_tensor, _ = train_dataset[i]
            label = label_tensor.item()
            label_counts[label] += 1

        if verbose == 1:
            print(f"Fold {fold+1} - Training set class distribution:")
            print(f"Class 0 (HC): {label_counts[0]}")
            print(f"Class 1 (FCD): {label_counts[1]}")

        # Conta le classi nel val set
        label_counts = {0: 0, 1: 0}
        for i in val_indices:
            _, label_tensor, _ = val_dataset[i]
            label = label_tensor.item()
            label_counts[label] += 1

        if verbose == 1: 
            print(f"Fold {fold+1} - Validation set class distribution:")
            print(f"Class 0 (HC): {label_counts[0]}")
            print(f"Class 1 (FCD): {label_counts[1]}")

        # Conta le classi nel test set
        label_counts = {0: 0, 1: 0}
        for i in test_idx:
            _, label_tensor, _ = test_dataset[i]
            label = label_tensor.item()
            label_counts[label] += 1

        if verbose == 1:
            print(f"Fold {fold+1} - Test set class distribution:")
            print(f"Class 0 (HC): {label_counts[0]}")
            print(f"Class 1 (FCD): {label_counts[1]}")


        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
        # Build model
        # TODO: check risoluzione modello originale chen per eventuale criticità diversa risuluzione features
        # TODO: batchnorm con batch piccolo tende ad essere rumoroso...ma in transfer ha i pesi pretrainati...valutiamo in seconda battuta se replace con groupnorm + ft
        model = resnet18(sample_input_D=181, sample_input_H=217, sample_input_W=181, num_seg_classes=1)

        print(model)
        summary(model, (1, 192, 256, 256))
        model.to(device)

        if verbose == 1:
        # Check GPU
            print(torch.cuda.is_available())  # Returns True if a GPU is detected
            print(torch.cuda.device_count())  # Number of available GPUs
            print(torch.cuda.get_device_name(0))  # Name of the first GPU
            print(torch.cuda.current_device())  # Index of the currently selected GPU

        net_dict = model.state_dict()
        
        pretrain = torch.load("resnet_18_23dataset.pth", map_location=device)
        pretrain_names = pretrain['state_dict']
        renamed_dict = {k.replace('module.', ''): v for k, v in pretrain_names.items()}
        pretrain_dict = {k: v for k, v in renamed_dict.items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        
        # Freezing dell'encoder (tutti i layer convoluzionali tranne layer4 e fc)
        for name, param in model.named_parameters():
            if not (name.startswith("fc") or name.startswith("layer4") or name.startswith("avgpool")):
                param.requires_grad = False 
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        torch.cuda.synchronize()

        # Directory di destinazione per salvare le immagini
        save_dir = r"C:\Users\ludov\Desktop\Results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Crea la cartella se non esiste

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
                    
                optimizer.zero_grad()
                outputs = model(images)

                # Calcola la perdita
                loss = criterion(outputs, labels.float())  # Squeeze per eliminare la dimensione 1, se necessario
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                # Aggiungi il print per mostrare l'output e l'ID del partecipante
                print(f"[TRAIN] Participant ID: {participant_id[0]} - Output: {outputs.detach().cpu().numpy()} - Loss: {loss.item():.4f}")

                print(f"Processing batch {idx+1}/{len(train_loader)}")
            
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)  # Aggiungi la loss di addestramento alla lista

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels, participant_id in val_loader:

                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
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

                # Salva il valore di fc.bias sulla GPU
                best_fc_bias = model.fc.bias.detach().clone()
                
                torch.save({'state_dict':model.state_dict()}, f"best_model_fold_{fold}.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # Crea una cartella per i grafici se non esiste
        os.makedirs('plots', exist_ok=True)

        # Al termine del training del fold, puoi tracciare le perdite per il fold corrente
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss per Epoch - Fold {fold + 1}')
        plt.legend()
        # Salva il grafico in un file (es. PNG)
        plt.savefig(f'plots/loss_fold_{fold + 1}.png', dpi=300)
        #plt.show() 

        # Aggiungi la fase di testing dopo il training del fold

        best_weights = torch.load(f"best_model_fold_{fold}.pth")
        best_names = best_weights['state_dict'] 
        model.load_state_dict(best_names)
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

                total += labels.size(0)  # Aggiungi il numero di esempi
                correct += (predicted == labels).sum().item()  # Conta i corretti

                print(f"[FOLD {fold}] Test Batch {batch_idx+1}/{len(test_loader)} - Participant: {participant_id[0]}")
                print(f"         Predicted: {predicted}, True: {labels}")

        accuracy = correct / total  # Calcola l'accuratezza per questo fold
        accuracies.append(accuracy)  # Aggiungi l'accuratezza di questo fold alla lista

        print(f"Fold {fold+1} Accuracy: {accuracy * 100:.8f}%")

        with open("accuracies.txt", "a") as f:
            f.write(f"[{experiment_name}] Fold {fold+1} | Acc: {accuracy * 100:.8f}% | LR: {lr} | BS: {batch_size} | Epochs: {num_epochs} | Patience: {patience}\n")


    # Calcola la media delle accuratezze
    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy across all folds: {mean_accuracy * 100:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--experiment_name", type=str, default="default_exp")
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        optimizer=args.optimizer,
        experiment_name=args.experiment_name
    )
