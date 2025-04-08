import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold
from resnet import resnet50  # Assicurati di avere resnet50
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchsummary import summary

# Carica il file Excel
excel_path = 'C:\\Users\\ludov\\Desktop\\OpenDataset\\participants.xlsx'
df = pd.read_excel(excel_path, usecols=['participant_id', 'group'])
label_mapping = {row['participant_id']: 1 if row['group'] == 'fcd' else 0 for _, row in df.iterrows()}

# Dataset NIfTI
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(192, 256, 256)):
        self.root_dir = root_dir
        self.label_mapping = label_mapping
        self.transform = transform
        self.target_size = target_size
        self.img_files = []
        
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if os.path.isdir(subject_path):
                anat_path = os.path.join(subject_path, "anat")
                if os.path.isdir(anat_path):
                    for file in os.listdir(anat_path):
                        if "T1w.nii" in file:
                            self.img_files.append(os.path.join(anat_path, file))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path)
        img = nib.load(img_path).get_fdata()
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)
        if self.transform:
            img = self.transform(img)
        label = self.label_mapping.get(img_name, 0)
        return img, torch.tensor(label, dtype=torch.long)

# Modello con pesi pre-addestrati
class MedicalNet(nn.Module):
    def __init__(self, path_to_weights, device):
        super(MedicalNet, self).__init__()
        self.model = resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=2)
        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.1)
        )
        net_dict = self.model.state_dict()
        pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
        pretrain_dict = {k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()}
        net_dict.update(pretrain_dict)
        self.model.load_state_dict(net_dict)
        self.fc = nn.Linear(512, 1)
        for param_name, param in self.model.named_parameters():
            param.requires_grad = param_name.startswith("conv_seg")
    
    def forward(self, x):
        features = self.model(x)
        return self.fc(features)

# Training parameters
num_epochs = 50
batch_size = 1
num_classes = 2
lr = 0.001
patience = 5
num_folds = 5

# Dataset e Cross Validation
root_dir = r'C:\\Users\\ludov\\Desktop\\OpenDataset'
dataset = NiftiDataset(root_dir=root_dir)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross Validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}/{num_folds}")
    train_size = int(len(train_idx) * 0.8)
    val_size = len(train_idx) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_idx, [train_size, val_size])

    train_loader = DataLoader(Subset(dataset, train_subset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_subset), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    # Inizializza il modello con pesi pre-addestrati
    model = MedicalNet(path_to_weights="resnet_50_23dataset.pth", device=device).to(device)
    summary(model, (1, 192, 256, 256))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
