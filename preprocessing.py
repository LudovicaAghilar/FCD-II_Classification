import os
import nibabel as nib
import numpy as np
import torch
from monai.transforms import RandRotate
import SimpleITK as sitk
from tqdm import tqdm

def crop_with_padding(img, padding=30):
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

    return img[min_x:max_x, min_y:max_y, min_z:max_z]

def rotate_and_crop(img, padding=30):
    cropped_img = crop_with_padding(img, padding)
    #rand_rotate = RandRotate(range_x=np.radians(15), range_y=np.radians(15), range_z=np.radians(15), prob=1.0, keep_size=True)
    return cropped_img

def register_image(fixed_path, moving_path):
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 100, relaxationFactor=0.9)
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform()))

    final_transform = registration_method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    return resampled

def preprocess_all_images(input_dir, output_dir, fixed_image_path):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith("T1w_brain.nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".nii.gz", "_preprocessed.nii.gz"))

            if os.path.exists(output_path):
                continue  # Skip already processed

            # Step 1: Registrazione
            registered_img = register_image(fixed_image_path, input_path)
            

            # Convert to numpy array for further processing
            img_array = sitk.GetArrayFromImage(registered_img)  # Z, Y, X
            print(f"{filename} - Shape: {img_array.shape}")

            # Step 2: Rotate and crop
            img_array = rotate_and_crop(img_array)

            # Converti in numpy se necessario
            if isinstance(img_array, torch.Tensor):
                img_array = img_array.numpy()

            # Converti l'array numpy in un'immagine SimpleITK
            cropped_resampled = sitk.GetImageFromArray(img_array)

            # Imposta lo spacing, origin e direction uguali all'immagine registrata
            cropped_resampled.SetSpacing(registered_img.GetSpacing())
            cropped_resampled.SetOrigin(registered_img.GetOrigin())
            cropped_resampled.SetDirection(registered_img.GetDirection())

            # Resample (ovvero riadatta le dimensioni dell'immagine)
            resampled_image = sitk.Resample(cropped_resampled, registered_img)

            # Stampa la dimensione finale
            print(f"{filename} - Shape after resampling: {sitk.GetArrayFromImage(resampled_image).shape}")

            # Step 3: Normalize
            img_array = sitk.GetArrayFromImage(resampled_image)
            img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)

            # Step 4: Save as new NIfTI
            processed_img = sitk.GetImageFromArray(img_array)
            processed_img.CopyInformation(registered_img)
            sitk.WriteImage(processed_img, output_path)
            print(f"Saved: {output_path}")

# USAGE
if __name__ == "__main__":
    input_dir = r"C:\Users\ludov\Desktop\Dataset_modified\hd_bet_output"
    output_dir = r"C:\Users\ludov\Desktop\Dataset_modified\preprocessed"
    fixed_image = r'C:\Users\ludov\Desktop\Dataset_modified\hd_bet_output\Template.nii'
    preprocess_all_images(input_dir, output_dir, fixed_image)
