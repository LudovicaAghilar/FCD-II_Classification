import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from resnet import resnet18  # Assicurati che il tuo script resnet.py contenga questa funzione
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import random
import torchio as tio

# Imposta il seed per la riproducibilità
""" def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per la modalità multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False """

#set_seed(42)

# Percorso al file Excel contenente le etichette
excel_path = 'C:\\Users\\ludov\\Desktop\\OpenDataset\\participants.xlsx'
df = pd.read_excel(excel_path, usecols=['participant_id', 'group'])

# Mappa filename → label numerico (hc → 0, fcd → 1)
label_mapping = {row['participant_id']: 1 if row['group'] == 'fcd' else 0 for _, row in df.iterrows()}

# Parametri di training
num_epochs = 50
batch_size = 8
lr = 0.001
patience_lr = 5
patience = 10
num_folds = 5
verbose = 0

# Carica il dataset
root_dir = r"C:\Users\ludov\Desktop\T1w_images"
#root_dir = r"C:\Users\ludov\Desktop\T1w_images_hd_bet"

# Ottieni tutte le etichette
all_files = [file for file in os.listdir(root_dir) if file.endswith("_T1w.nii.gz")]
all_labels = [label_mapping[file.split('_')[0]] for file in all_files]
#all_labels = [label_mapping[file.split('_')[1]] for file in all_files]

# >>> QUI: stampa dimensioni originali
#print("Dimensioni originali dei volumi:")
""" for file in all_files:
    img_path = os.path.join(root_dir, file)
    img_nib = nib.load(img_path)
    data = img_nib.get_fdata()
    print(f"{file}: {data.shape}")  """

""" # Trasformazione di preprocessing unica: normalizza dimensioni """
preprocess_transform = tio.Compose([
    tio.Resample(
        target=(1.0, 1.0, 1.0),                # nuovo spacing isotropico in mm
        image_interpolation='linear',         # per immagini continue
    ),
    tio.CropOrPad((160, 256, 256)),  # Ridimensiona tutte le immagini a (160, 256, 256)
    tio.ZNormalization(),
])

root_img = r"C:\Users\ludov\Scripts"
# Preprocessing di tutte le immagini prima dello split
output_dir = os.path.join(root_img, 'preprocessed_T1w')  # Dove salvare le immagini trasformate
os.makedirs(output_dir, exist_ok=True)

# Preprocessing di tutte le immagini prima dello split
all_subjects = []
for file in all_files:
    img_path = os.path.join(root_dir, file)
    participant_id = file.split('_')[0]
    label = label_mapping.get(participant_id, 0)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        label=torch.tensor(label, dtype=torch.long),
        participant_id=participant_id
    )
    # Applica trasformazione fissa

    subject = preprocess_transform(subject)
    all_subjects.append(subject)

    # Salva immagine trasformata
    output_img_path = os.path.join(output_dir, f"{participant_id}_preprocessed.nii.gz")
    subject['image'].save(output_img_path)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import train_test_split

# Split iniziale 80% train_val e 20% test
train_val_subjects, test_subjects, train_val_labels, test_labels = train_test_split(
    all_subjects, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

#Crea test dataset e loader UNA VOLTA
test_dataset = tio.SubjectsDataset(test_subjects)
test_loader = tio.SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_subjects, train_val_labels)):
    print(f"Training fold {fold+1}/{num_folds}")

    train_subjects = [train_val_subjects[i] for i in train_idx]
    val_subjects = [train_val_subjects[i] for i in val_idx]

    train_labels = [train_val_labels[i] for i in train_idx]

    # Salva i nomi dei file usati per training e validation
    train_ids = [subject['participant_id'] for subject in train_subjects]
    val_ids = [subject['participant_id'] for subject in val_subjects]
    
    split_save_path = f"split_fold_{fold + 1}.txt"
    with open(split_save_path, "w") as f:
        f.write("TRAINING SET:\n")
        for pid in train_ids:
            f.write(f"{pid}\n")
        
        f.write("\nVALIDATION SET:\n")
        for pid in val_ids:
            f.write(f"{pid}\n")

    # Augmentation solo sul train
    train_transform = tio.RandomAffine(degrees=15)

    # Dataset
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    val_dataset = tio.SubjectsDataset(val_subjects)

    # Loader
    train_loader = tio.SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = tio.SubjectsLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Modello
    model = resnet18(sample_input_D=160, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
    model.to(device)

    #Caricamento pesi pre-addestrati
    net_dict = model.state_dict()
    pretrain = torch.load("resnet_18_23dataset.pth", map_location=device)
    pretrain_names = pretrain['state_dict']
    renamed_dict = {k.replace('module.', ''): v for k, v in pretrain_names.items()}
    pretrain_dict = {k: v for k, v in renamed_dict.items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)

    # Freezing parziale
    for name, param in model.named_parameters():
        #print(name)
        if not (name.startswith("fc") or name.startswith("layer4") or name.startswith("avgpool")):
            param.requires_grad = False
    
    import re
 
    # Match 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias'
    pattern = re.compile(r'^layer4\.[01]\.bn[12]\.(weight|bias)$')

    # Assuming 'model' is your PyTorch model
    for name, param in model.named_parameters():
        if pattern.match(name):
            param.requires_grad = False
            #print(f"Disabled gradient for: {name}")
    
    summary(model,(1,160,256,256))

    # Ottimizzazione
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_lr)

    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    torch.cuda.synchronize()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'][tio.DATA].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)
            #print(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #participant_id = batch['participant_id']

             # Aggiungi il print per mostrare l'output e l'ID del partecipante
            #print(f"[TRAIN] Participant ID: {participant_id[0]} - Output: {outputs.detach().cpu().numpy()} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'][tio.DATA].to(device)
                labels = batch['label'].to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

    best_weights = torch.load(f"best_model_fold_{fold}.pth")
    best_names = best_weights['state_dict'] 
    model.load_state_dict(best_names)
    model.eval()
    correct = 0
    total = 0

    # TEST: predizione e salvataggio
    test_pred_file = f"test_predictions_fold_{fold}.txt"
    with open(test_pred_file, "w") as f:
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'][tio.DATA].to(device)
                labels = batch['label'].to(device).float().unsqueeze(1)
                participant_ids = batch['participant_id']

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                predicted = probs.round()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                for pid, true_label, prob, pred in zip(participant_ids, labels, probs, predicted):
                    f.write(f"{pid}, True: {true_label.item()}, Pred: {pred.item()}, Prob: {prob.item():.4f}\n")
    
    accuracy = correct / total
    accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy * 100:.8f}%")

    with open("accuracies.txt", "a") as f:
        f.write(f"Fold {fold+1} Accuracy: {accuracy * 100:.8f}%\n")

# Accuracy media
mean_accuracy = np.mean(accuracies)
print(f"Average Accuracy across {num_folds} folds: {mean_accuracy:.4f}")