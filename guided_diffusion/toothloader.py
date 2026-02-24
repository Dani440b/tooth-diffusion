import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import pandas as pd

class ToothVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, metadata_path, test_flag=False, normalize=None, mode='train', img_size=256):
        super().__init__()
        self.mode = mode

        self.directory = os.path.expanduser(directory)
        self.metadata_path = os.path.expanduser(metadata_path)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []
        
        self.image_dir = os.path.join(directory, "Images")
        self.label_dir = os.path.join(directory, "Labels")
        
        #Check for meta data
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.csv no found at {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata.columns = self.metadata.columns.str.strip()  # removes leading/trailing spaces
     
        for f in os.listdir(self.image_dir):
            if not f.endswith(".nii.gz"):
                continue
            # Some image files have a suffix like '_0000.nii.gz' while label files use the basename without that suffix.
            image_fname = f
            if f.endswith('_0000.nii.gz'):
                label_fname = f.replace('_0000.nii.gz', '.nii.gz')
            else:
                label_fname = f
            label_path = os.path.join(self.label_dir, label_fname)
            image_path = os.path.join(self.image_dir, image_fname)
            if os.path.exists(label_path):
                self.database.append({
                    "image": image_path,
                    "label": label_path,
                    "name": image_fname
                })
        print(f"Found {len(self.database)} samples in '{mode}' set.")
      
    #Normalize
    def normalize_image(self, x):
        return ((x - x.min()) / (x.max() - x.min()))
    
    def __getitem__(self, x):
        filedict = self.database[x]
        
        image_np = (nibabel.as_closest_canonical(nibabel.load(filedict["image"]))).get_fdata(dtype=np.float32)
        label_np = (nibabel.as_closest_canonical(nibabel.load(filedict["label"]))).get_fdata(dtype=np.float32)

        image_np = self.normalize_image(image_np)
        label_np = label_np.astype(np.uint8)
        
        # Create brain mask: anything with label > 0 is brain
        brain_mask = (label_np > 0).astype(np.float32)
        
        # Keep full image with all slices (no cropping)
        # We filter out volumes with empty masks at load time, but use full unmasked images for training

        # Name and conditions
        filename = os.path.basename(filedict['name'])
        basename = filename
        # Extract basename without extension and suffix (e.g., 002_S_0295_I45108 from 002_S_0295_I45108_0000.nii.gz)
        if filename.endswith('_0000.nii.gz'):
            basename_without_ext = filename.replace('.nii.gz', '').rsplit('_', 1)[0]
        else:
            basename_without_ext = filename.replace('.nii.gz', '')
        row_metadata = self.metadata[self.metadata['BASENAME'] == basename_without_ext]
        if row_metadata.empty:
            raise KeyError(f"{basename} not found in metadata")
        
        row_metadata = row_metadata.iloc[0] # Convert to Series
        
        # Vector 1: Diagnosis (one-hot encoding)
        diagnosis_mapping = {'CN': 0, 'MCI': 1, 'AD': 2}  # CN, MCI, AD are common in ADNI
        diagnosis = row_metadata.get('Screen.Diagnosis', 'CN')
        diagnosis_onehot = torch.zeros(3, dtype=torch.float32)
        diagnosis_onehot[diagnosis_mapping.get(diagnosis, 0)] = 1.0
        
        # Vector 3: Age (normalize to [0, 1])
        age = float(row_metadata.get('Age', 75)) / 100.0  # Assume age range 0-100
        age_tensor = torch.tensor([age], dtype=torch.float32)
        
        # Vector 4: Sex (0=F, 1=M)
        sex = row_metadata.get('Sex', 'M')
        sex_tensor = torch.tensor([1.0 if sex == 'M' else 0.0], dtype=torch.float32)
        
        vectors = {
            "diagnosis": diagnosis_onehot,
            "age": age_tensor,
            "sex": sex_tensor,
            "brain_mask": torch.tensor(brain_mask, dtype=torch.float32),
        }

        
        # Make copy to ensure that it doesn't modify each other
        # Use full unmasked image for training
        target_image_np = np.ascontiguousarray(image_np)
        target_label_np = np.ascontiguousarray(brain_mask)
        cond_image_np = np.ascontiguousarray(image_np)
        cond_label_np = np.ascontiguousarray(brain_mask)

        # For MRI, skip tooth-specific augmentation (no teeth in MRI)
        # The augment_missing_teeth and reconstruct_3_mode flags are for tooth data only

        if not self.mode == 'fake':
            image = torch.from_numpy(target_image_np).unsqueeze(0)
            label = torch.from_numpy(target_label_np).unsqueeze(0).long()
            cond_image = torch.from_numpy(cond_image_np).unsqueeze(0)
            cond_label = torch.from_numpy(cond_label_np).unsqueeze(0).long()
        else:
            image = torch.from_numpy(np.ascontiguousarray(image_np)).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray(label_np)).unsqueeze(0).long()
            cond_image = torch.from_numpy(np.ascontiguousarray(cond_image_np)).unsqueeze(0)
            cond_label = torch.from_numpy(np.ascontiguousarray(cond_label_np)).unsqueeze(0).long()

        image = self.normalize(image)
        cond_image = self.normalize(cond_image)     
        
        # Adding the random flipping of image and label       
        if self.mode == 'train' and np.random.rand() < 0.5:
            # flip horizontally // mirror
            image = torch.flip(image, dims=[1])
            cond_image = torch.flip(cond_image, dims=[1])
            label = torch.flip(label, dims=[1])
            cond_label = torch.flip(cond_label, dims=[1])

            upper = (label > 0) & (label <= 16)
            lower = (label >= 17) & (label <= 32)

            label[upper] = 17 - label[upper]
            label[lower] = 49 - label[lower]

            cond_label[upper] = 17 - cond_label[upper]
            cond_label[lower] = 49 - cond_label[lower]
                        
        if self.mode in ['eval', 'fake']:
            return {
                "image": image,
                "label": label,
                "cond_image": cond_image,
                "cond_label": cond_label,
                "name": [basename],
                "diagnosis": vectors["diagnosis"],
                "age": vectors["age"],
                "sex": vectors["sex"],
                "brain_mask": vectors.get("brain_mask", torch.ones_like(label)),
            }
        return {
            "image": image,
            "label": label,
            "cond_image": cond_image,
            "cond_label": cond_label,
            "diagnosis": vectors["diagnosis"],
            "age": vectors["age"],
            "sex": vectors["sex"],
            "brain_mask": vectors.get("brain_mask", torch.ones_like(label)),
        }
    
    def __len__(self):
        return len(self.database)