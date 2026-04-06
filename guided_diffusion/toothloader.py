import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
import json

class ToothVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, metadata_path, test_flag=False, normalize=None, mode='train', img_size=256,
                 noisy_dir=None, noisy_meta_data=None):
        super().__init__()
        self.mode = mode

        self.directory = os.path.expanduser(directory)
        self.metadata_path = os.path.expanduser(metadata_path)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []
        self.noisy_dir = os.path.expanduser(noisy_dir) if noisy_dir else None
        self.quality_dim = 7  # 7 artifact metrics only
        
        self.image_dir = os.path.join(directory, "Images")
        self.label_dir = os.path.join(directory, "Labels")
        
        #Check for meta data
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.csv no found at {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata.columns = self.metadata.columns.str.strip()  # removes leading/trailing spaces

        # Fast lookup by BASENAME for clinical metadata joins.
        self._metadata_by_basename = self.metadata.set_index('BASENAME') if 'BASENAME' in self.metadata.columns else None

        if self.noisy_dir is not None:
            # DCP-Diff style: load paired (clean, noisy) entries from augmentation_metadata.csv.
            # clean image  -> diffusion target (x_0)
            # noisy image  -> conditioning guide (r), concatenated to x_t at every timestep
            noisy_meta_path = noisy_meta_data or os.path.join(self.noisy_dir, "augmentation_metadata.csv")
            if not os.path.exists(noisy_meta_path):
                raise FileNotFoundError(f"augmentation_metadata.csv not found at {noisy_meta_path}")
            aug_meta = pd.read_csv(noisy_meta_path)
            aug_meta.columns = aug_meta.columns.str.strip()
            noisy_image_dir = os.path.join(self.noisy_dir, "Images")
            clean_aligned_dir = os.path.join(self.noisy_dir, "Clean")
            for _, row in aug_meta.iterrows():
                clean_fname = row['clean_image']
                noisy_fname = row['noisy_image']
                # Use geometry-aligned clean from augmented dir (same filename as noisy)
                clean_path = os.path.join(clean_aligned_dir, noisy_fname)
                if not os.path.exists(clean_path):
                    # Fallback to original clean if aligned version not found
                    clean_path = os.path.join(self.image_dir, clean_fname)
                noisy_path = os.path.join(noisy_image_dir, noisy_fname)
                if clean_fname.endswith('_0000.nii.gz'):
                    label_fname = clean_fname.replace('_0000.nii.gz', '.nii.gz')
                else:
                    label_fname = clean_fname
                label_path = os.path.join(self.label_dir, label_fname)
                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    self.database.append({
                        "image": clean_path,
                        "label": label_path if os.path.exists(label_path) else None,
                        "cond_image": noisy_path,
                        "name": clean_fname,
                        "aug_row": row.to_dict(),
                    })
            print(f"Found {len(self.database)} noisy/clean paired samples in '{mode}' set.")
        else:
            # Original behaviour: scan Images dir, cond_image defaults to clean image.
            for f in os.listdir(self.image_dir):
                if not f.endswith(".nii.gz"):
                    continue
                # Some image files have a suffix like '_0000.nii.gz' while label files use the basenaame without that suffix.
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
                        "cond_image": None,  # will fall back to clean image
                        "name": image_fname,
                        "aug_row": None,
                    })
            print(f"Found {len(self.database)} samples in '{mode}' set.")

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _extract_transform_strength(self, params, key, default=0.0):
        value = params.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return self._safe_float(np.mean(value), default=default)
        return self._safe_float(value, default=default)

    def _quality_from_applied_transforms(self, applied_transforms_raw):
        # Quality order: [sdr, cvr, truncation_ratio, alpha, psnr_normalized, freq_ratio, spike_fraction]
        quality = np.zeros(self.quality_dim, dtype=np.float32)
        if not applied_transforms_raw:
            return quality

        try:
            if isinstance(applied_transforms_raw, str):
                transforms = json.loads(applied_transforms_raw)
            elif isinstance(applied_transforms_raw, dict):
                transforms = applied_transforms_raw
            else:
                transforms = {}
        except Exception:
            transforms = {}

        sdr = 1.0
        cvr = 1.0
        truncation_ratio = 1.0
        alpha = 1.0
        psnr_normalized = 1.0
        freq_ratio = 1.0
        spike_fraction = 0.0

        if "RandomNoise" in transforms:
            p = transforms.get("RandomNoise", {})
            std_val = self._extract_transform_strength(p, "std", default=0.0)
            if std_val == 0.0:
                std_val = self._extract_transform_strength(p, "std_range", default=0.0)
            psnr_normalized = float(np.clip(1.0 - (std_val / 0.08), 0.0, 1.0))

        if "RandomSpike" in transforms:
            p = transforms.get("RandomSpike", {})
            num_spikes = self._extract_transform_strength(p, "num_spikes", default=0.0)
            if num_spikes == 0.0:
                num_spikes = self._extract_transform_strength(p, "num_spikes_range", default=0.0)
            spike_fraction = float(np.clip(num_spikes / 5.0, 0.0, 1.0))

        if "RandomBlur" in transforms:
            p = transforms.get("RandomBlur", {})
            sigma = self._extract_transform_strength(p, "std", default=0.0)
            if sigma == 0.0:
                sigma = self._extract_transform_strength(p, "std_range", default=0.0)
            freq_ratio = float(np.clip(1.0 - (sigma / 1.0), 0.0, 1.0))

        if "RandomBiasField" in transforms:
            p = transforms.get("RandomBiasField", {})
            coeff = self._extract_transform_strength(p, "coefficients", default=0.0)
            cvr = float(np.clip(1.0 - (coeff / 0.5), 0.0, 1.0))

        if "RandomGamma" in transforms:
            p = transforms.get("RandomGamma", {})
            log_gamma = self._extract_transform_strength(p, "log_gamma", default=0.0)
            if log_gamma == 0.0:
                log_gamma = self._extract_transform_strength(p, "log_gamma_range", default=0.0)
            sdr = float(np.clip(1.0 - abs(log_gamma) / 0.3, 0.0, 1.0))

        if "RandomSwap" in transforms:
            p = transforms.get("RandomSwap", {})
            patch = self._extract_transform_strength(p, "patch_size", default=0.0)
            truncation_ratio = float(np.clip(1.0 - (patch / 15.0), 0.0, 1.0))

        if "RandomGhosting" in transforms:
            p = transforms.get("RandomGhosting", {})
            intensity = self._extract_transform_strength(p, "intensity", default=0.0)
            if intensity == 0.0:
                intensity = self._extract_transform_strength(p, "intensity_range", default=0.0)
            alpha = float(np.clip(1.0 - (intensity / 0.7), 0.0, 1.0))

        if "RandomMotion" in transforms:
            p = transforms.get("RandomMotion", {})
            degrees = self._extract_transform_strength(p, "degrees", default=0.0)
            translation = self._extract_transform_strength(p, "translation", default=0.0)
            motion_alpha = 1.0 - np.clip((degrees / 10.0 + translation / 5.0) / 2.0, 0.0, 1.0)
            alpha = float(min(alpha, motion_alpha))

        quality[:7] = np.array(
            [sdr, cvr, truncation_ratio, alpha, psnr_normalized, freq_ratio, spike_fraction],
            dtype=np.float32,
        )
        return quality

    def _extract_quality_vector(self, aug_row):
        # Preferred path: explicit quality columns in future metadata schema.
        if isinstance(aug_row, dict):
            canonical_artifact_keys = [
                "sdr",
                "cvr",
                "truncation_ratio",
                "alpha",
                "psnr_normalized",
                "freq_ratio",
                "spike_score",
            ]
            if all(k in aug_row for k in canonical_artifact_keys):
                artifacts = [self._safe_float(aug_row.get(k, 0.0), default=0.0) for k in canonical_artifact_keys]
                artifacts = np.clip(np.array(artifacts, dtype=np.float32), 0.0, 1.0)
                return artifacts

            prefixed = [k for k in aug_row.keys() if str(k).startswith("quality_")]
            prefixed = [k for k in prefixed if str(k) not in ("quality_overall", "quality_overall_score")]
            if len(prefixed) >= self.quality_dim:
                prefixed = sorted(prefixed)
                values = [self._safe_float(aug_row[k], default=0.0) for k in prefixed[:self.quality_dim]]
                return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)

            if all(k in aug_row for k in ["q_noise", "q_spike", "q_blur", "q_bias", "q_ghost", "q_motion", "q_overall"]):
                # Legacy 6+overall format: map to 7 artifact-only values by repeating motion score for the 7th slot.
                values = [
                    self._safe_float(aug_row.get("q_noise", 0.0)),
                    self._safe_float(aug_row.get("q_spike", 0.0)),
                    self._safe_float(aug_row.get("q_blur", 0.0)),
                    self._safe_float(aug_row.get("q_bias", 0.0)),
                    self._safe_float(aug_row.get("q_ghost", 0.0)),
                    self._safe_float(aug_row.get("q_motion", 0.0)),
                    self._safe_float(aug_row.get("q_motion", 0.0)),
                ]
                return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)

            return self._quality_from_applied_transforms(aug_row.get("applied_transforms", ""))

        return np.zeros(self.quality_dim, dtype=np.float32)
      
    #Normalize
    def normalize_image(self, x):
        return ((x - x.min()) / (x.max() - x.min()))
    
    def __getitem__(self, x):
        filedict = self.database[x]
        
        image_np = (nibabel.as_closest_canonical(nibabel.load(filedict["image"]))).get_fdata(dtype=np.float32)

        if filedict.get("label") is not None:
            label_np = (nibabel.as_closest_canonical(nibabel.load(filedict["label"]))).get_fdata(dtype=np.float32)
            label_np = label_np.astype(np.uint8)
            brain_mask = (label_np > 0).astype(np.float32)
        else:
            label_np = np.zeros_like(image_np, dtype=np.uint8)
            brain_mask = np.zeros_like(image_np, dtype=np.float32)

        image_np = self.normalize_image(image_np)
        
        # Load noisy conditioning image if a paired noisy path is provided (DCP-Diff style).
        # Otherwise fall back to using the clean image as its own condition.
        if filedict.get("cond_image") is not None:
            cond_image_np = (nibabel.as_closest_canonical(nibabel.load(filedict["cond_image"]))).get_fdata(dtype=np.float32)
            cond_image_np = self.normalize_image(cond_image_np)
            self._paired_noisy = True  # flag: skip extra random flips on pre-augmented data
        else:
            cond_image_np = np.ascontiguousarray(image_np)
            self._paired_noisy = False
        
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
        if self._metadata_by_basename is None or basename_without_ext not in self._metadata_by_basename.index:
            raise KeyError(f"{basename} not found in metadata")

        row_metadata = self._metadata_by_basename.loc[basename_without_ext]
        if isinstance(row_metadata, pd.DataFrame):
            row_metadata = row_metadata.iloc[0]
        
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

        quality_np = self._extract_quality_vector(filedict.get("aug_row"))
        quality_tensor = torch.tensor(quality_np, dtype=torch.float32)

        metadata_cond = torch.cat([diagnosis_onehot, age_tensor, sex_tensor, quality_tensor], dim=0)
        
        vectors = {
            "diagnosis": diagnosis_onehot,
            "age": age_tensor,
            "sex": sex_tensor,
            "quality": quality_tensor,
            "metadata_cond": metadata_cond,
            "brain_mask": torch.tensor(brain_mask, dtype=torch.float32),
        }

        
        # Make copy to ensure that it doesn't modify each other
        # Use full unmasked image for training
        target_image_np = np.ascontiguousarray(image_np)
        target_label_np = np.ascontiguousarray(brain_mask)
        cond_image_np = np.ascontiguousarray(cond_image_np)
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
            label = torch.from_numpy(np.ascontiguousarray(brain_mask)).unsqueeze(0).long()
            cond_image = torch.from_numpy(np.ascontiguousarray(cond_image_np)).unsqueeze(0)
            cond_label = torch.from_numpy(np.ascontiguousarray(brain_mask)).unsqueeze(0).long()

        image = self.normalize(image)
        cond_image = self.normalize(cond_image)

        # Random flip for augmentation.
        # When using pre-augmented noisy images (noisy_dir mode), flipping is skipped because
        # the noisy volume has already been spatially augmented independently of the clean image.
        if self.mode == 'train' and not self._paired_noisy and np.random.rand() < 0.5:
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
                "quality": vectors["quality"],
                "metadata_cond": vectors["metadata_cond"],
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
            "quality": vectors["quality"],
            "metadata_cond": vectors["metadata_cond"],
            "brain_mask": vectors.get("brain_mask", torch.ones_like(label)),
        }
    
    def __len__(self):
        return len(self.database)