import SimpleITK as sitk
import numpy as np
from monai.transforms import RandRotate

# Funzione di crop con padding (già presente nel tuo codice)
def crop_with_padding(img, padding=20):
    non_zero_indices = np.where(img > 0)  
    min_x, max_x = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_z, max_z = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    min_x = max(min_x - padding, 0)
    max_x = min(max_x + padding, img.shape[0])
    min_y = max(min_y - padding, 0)
    max_y = min(max_y + padding, img.shape[1])
    min_z = max(min_z - padding, 0)
    max_z = min(max_z + padding, img.shape[2])

    cropped_img = img[min_x:max_x, min_y:max_y, min_z:max_z]
    return cropped_img

# Percorsi delle immagini
fixed = sitk.ReadImage(r'C:\Users\ludov\Desktop\Dataset_modified\hd_bet_output\Template.nii', sitk.sitkFloat32)
moving = sitk.ReadImage(r'C:\Users\ludov\Desktop\Dataset_modified\hd_bet_output\sub-00015_acq-iso08_T1w_brain.nii.gz', sitk.sitkFloat32)

# Inizializzazione del metodo di registrazione
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation()

# Configurazione dell'ottimizzatore
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=1.0, 
    minStep=0.001, 
    numberOfIterations=100, 
    relaxationFactor=0.9
)

# Imposta la trasformazione iniziale
registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform()))

# Esegui la registrazione
final_transform = registration_method.Execute(fixed, moving)

# Resample l'immagine in movimento (allineata)
resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

# Converti l'immagine registrata a numpy array per applicare il crop
resampled_array = sitk.GetArrayFromImage(resampled)

# Applica il cropping con padding
cropped_resampled_array = crop_with_padding(resampled_array, padding=5)

# Converti di nuovo in SimpleITK Image
cropped_resampled = sitk.GetImageFromArray(cropped_resampled_array)

# Inizializzazione della trasformazione RandRotate di MONAI (±15 gradi)
rand_rotate = RandRotate(range_x=np.radians(15), range_y=np.radians(15), range_z=np.radians(15), prob=1.0, keep_size=True)

# Applicazione della rotazione casuale usando RandRotate
cropped_resampled_array = rand_rotate(cropped_resampled_array)

# Converti di nuovo in SimpleITK Image
cropped_resampled = sitk.GetImageFromArray(cropped_resampled_array)

# Salva l'immagine ruotata
sitk.WriteImage(cropped_resampled, r'C:\Users\ludov\Desktop\Dataset_modified\hd_bet_output\subject_in_mni_00015_rotated.nii.gz')

print("Registrazione, cropping e rotazione completati, immagine salvata.")
