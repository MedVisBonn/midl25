"""
This script collects the Dataset classes for the PMRI and MNMv2 datasets.

Usage: serves only as a collection of individual functionalities
Authors: Jonathan Lennartz
"""

# - standard packages
from pathlib import Path
import os
# - third party packages
import pandas as pd
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop
import nibabel as nib


class PMRIDataset(Dataset):
    """
    Multi-site dataset for prostate MRI segmentation from:
    https://liuquande.github.io/SAML/
    Possible domains:
        ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
    Initialization parameters:
    - data_dir -> Path to dataset directory
    - vendor -> Vendor from possible ones to load data
    """

    _DS_CONFIG = {"num_classes": 2, "spatial_dims": 2, "size": (384, 384)}

    def __init__(
        self,
        data_dir: str,
        domain: str, # in ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"],
        non_empty_target: bool = True,
        normalize: bool = True,
    ):
        assert domain in ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"], "Invalid domain"
        self.domain = domain
        self._data_dir = Path(data_dir).resolve()
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self.target_spacing = 0.5
        self._crop = CenterCrop(384)
        self._load_data()


    def _load_data(self):
        self.input = []
        self.target = []

        site_path = self._data_dir / self.domain
        for file in site_path.iterdir():
            # Load the ones that have a segmentation associated file to them
            if "segmentation" in file.name.lower():
                case = file.name[4:6]
                seg_name = "Segmentation" if self.domain == "BMC" else "segmentation"
                case_input_path = site_path / f"Case{case}.nii.gz"
                case_target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
                x = nib.load(case_input_path)
                y = nib.load(case_target_path)

                spacing = x.header.get_zooms()[0]
                x = torch.tensor(x.get_fdata())
                y = torch.tensor(y.get_fdata(), dtype=torch.long)

                scale_factor = (1 / self.target_spacing * spacing, 1 / self.target_spacing * spacing)
                x = interpolate(
                    x.unsqueeze(1), 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(1)
                y = interpolate(
                    y.unsqueeze(1).float(), 
                    scale_factor=scale_factor, 
                    mode='nearest'
                ).long().squeeze(1)

                x = self._crop(x)
                y = self._crop(y)

                self.input.append(x)
                self.target.append(y)


        # Concatenate / Reshape to batch first / Add channel Axis
        self.input = torch.cat(self.input, dim=-1).moveaxis(-1, 0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=-1).moveaxis(-1, 0).unsqueeze(1)
        # Relabel cases if there are two prostate classes (Since not all datasets distinguish between the two)
        self.target[self.target == 2] = 1

        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]


        if self._normalize:
            mean = self.input.mean()
            std = self.input.std()
            self.input = (self.input - mean) / std


    def random_split(
        self,
        val_size: float = 0.2,
    ):
        class PMRISubset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]
            
            def __getitem__(self, idx):
                return {
                    "input": self.input[idx], 
                    "target": self.target[idx],
                    "index": idx
                }
            
        torch.manual_seed(0)
        indices = torch.randperm(len(self.input)).tolist()
        pmri_train = PMRISubset(
            input=self.input[indices[int(val_size * len(self.input)):]],
            target=self.target[indices[int(val_size * len(self.input)):]],
        )

        pmri_val = PMRISubset(
            input=self.input[indices[:int(val_size * len(self.input))]],
            target=self.target[indices[:int(val_size * len(self.input))]],
        )

        return pmri_train, pmri_val


    def __len__(self):
        return self.input.shape[0]
    

    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
            "index": idx
        }
    


class MNMv2Dataset(Dataset):
    """
    Vendors: 
        Siemens:
            Domain: Symphony, Number of cases: 4024
            Domain: Trio, Number of cases: 128
            Domain: Avanto, Number of cases: 904 
                
        GE (Signa): 
            Domain: HDxt, Number of cases: 618
            Domain: EXCITE, Number of cases: 632
            Domain: Explorer, Number of cases: 26

        Philips:
            Domain: Achieva, Number of cases: 1796 

        As List: ["Symphony", "Trio", "Avanto", "HDxt", "EXCITE", "Explorer", "Achieva"]    
    """
    def __init__(
        self,
        data_dir,
        domain,
        binary_target: str = False,
        non_empty_target: str = True,
        normalize: str = True,
    ):
        self.domain = domain
        self._data_dir = Path(data_dir).resolve()
        self._binary_target = binary_target
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._data_info = pd.read_csv(
            self._data_dir / "dataset_information.csv", index_col=0
        )
        self._crop = CenterCrop(256)
        self.target_spacing = 1.
        self._load_data()


    def _load_data(self):
        self.input = []
        self.target = []
        self.meta = []
        for case_ in self._data_info.index:
            if self.domain.lower() in self._data_info.loc[case_].SCANNER.lower():
                case_path = self._data_dir / "dataset" / f"{case_:03d}"
                modes = ["ES", "ED"]

                for mode in modes:
                    x = nib.load(case_path / f"{case_:03d}_SA_{mode}.nii.gz")
                    y = nib.load(case_path / f"{case_:03d}_SA_{mode}_gt.nii.gz")
                    spacing = x.header.get_zooms()[0]
                    x = torch.tensor(x.get_fdata()).moveaxis(-1, 0)
                    y = torch.tensor(y.get_fdata().astype(int), dtype=torch.long).moveaxis(-1, 0)
                    # interpolate
                    scale_factor = (1 / self.target_spacing * spacing, 1 / self.target_spacing * spacing)
                    x = interpolate(
                        x.unsqueeze(1), 
                        scale_factor=scale_factor, 
                        mode='bilinear', 
                        align_corners=True
                    ).squeeze(1)
                    y = interpolate(
                        y.unsqueeze(1).float(), 
                        scale_factor=scale_factor, 
                        mode='nearest'
                    ).long().squeeze(1)

                    x = self._crop(x)
                    y = self._crop(y)
                    self.input.append(x)
                    self.target.append(y)

        self.input  = torch.cat(self.input,  dim=0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=0).unsqueeze(1)

        self.target[self.target < 0] = 0

        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]

        if self._binary_target:
            self.target[self.target != 0] = 1

        if self._normalize:
            mean = self.input.mean()
            std = self.input.std()
            self.input = (self.input - mean) / std


    def random_split(
        self,
        val_size: float = 0.2,
        test_size: float = None,
    ):
        class MNMv2Subset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]
            
            def __getitem__(self, idx):
                return {
                    "input": self.input[idx], 
                    "target": self.target[idx],
                    "index": idx
                }
        

        torch.manual_seed(0)
        indices = torch.randperm(len(self.input)).tolist()

        if test_size is not None:

            test_split = int(test_size * len(self.input))
            val_split = int(val_size * len(self.input)) + test_split

            mnmv2_test = MNMv2Subset(
                input=self.input[indices[:test_split]],
                target=self.target[indices[:test_split]],
            )

            mnmv2_val = MNMv2Subset(
                input=self.input[indices[test_split:val_split]],
                target=self.target[indices[test_split:val_split]],
            )

            mnmv2_train = MNMv2Subset(
                input=self.input[indices[val_split:]],
                target=self.target[indices[val_split:]],
            )


            return mnmv2_train, mnmv2_val, mnmv2_test
        
        mnmv2_train = MNMv2Subset(
            input=self.input[indices[int(val_size * len(self.input)):]],
            target=self.target[indices[int(val_size * len(self.input)):]],
        )

        mnmv2_val = MNMv2Subset(
            input=self.input[indices[:int(val_size * len(self.input))]],
            target=self.target[indices[:int(val_size * len(self.input))]],
        )

        return mnmv2_train, mnmv2_val


    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
            "index": idx
        }