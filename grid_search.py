import itertools
import subprocess
import csv
import os
import sys

# Definizione dei parametri da esplorare
lr_values = [0.0001, 0.001]
batch_sizes = [8, 16]
optimizers = ['adam']
patience_values = [5, 15, 200]
epochs = [20, 50, 100]

# Lista per memorizzare i risultati
results = []

# Tutte le combinazioni possibili
combinations = list(itertools.product(lr_values, batch_sizes, optimizers, patience_values, epochs))

# File per salvare i risultati
output_file = 'hyperparameter_results.csv'
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lr', 'batch_size', 'optimizer', 'patience', 'epochs', 'mean_accuracy'])

for i, (lr, batch_size, optimizer, patience, epoch_val) in enumerate(combinations):
    experiment_name = f"exp_{i+1}"
    print(f"===> Running {experiment_name} with lr={lr}, bs={batch_size}, opt={optimizer}, patience={patience}, epochs={epoch_val}")

    python_executable = sys.executable  # Questo prende il path dell'interprete in uso

    # Esegui il training chiamando lo script principale
    result = subprocess.run(
        [
            python_executable, "Script_original.py", 
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--optimizer", optimizer,
            "--patience", str(patience),
            "--epochs", str(epoch_val),
            "--experiment_name", experiment_name
        ],
        capture_output=True,
        text=True
    )

    # Estrai l'accuratezza media dallo stdout (assumendo che venga stampata alla fine)
    output = result.stdout
    mean_acc_line = [line for line in output.splitlines() if "Mean Accuracy" in line]
    if mean_acc_line:
        mean_acc = float(mean_acc_line[0].split(":")[1].strip().replace('%', ''))
    else:
        mean_acc = 0.0  # fallback

    results.append((lr, batch_size, optimizer, patience, mean_acc))

    # Salva anche riga per riga sul CSV
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([lr, batch_size, optimizer, patience, epoch_val, mean_acc])

