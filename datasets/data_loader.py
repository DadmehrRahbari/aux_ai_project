# aux_ai_project/datasets/data_loader.py
import zipfile
import io
import params
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

class CityscapesZipLoader:
    def __init__(self, mode='val'):
        self.img_zip_path = params.CITYSCAPES_ZIPS["images"]
        self.label_zip_path = params.CITYSCAPES_ZIPS["labels"]
        self.mode = mode 
        
    def stream_images(self):
        """Yields (Image, GT_Mask, Path) directly from synchronized ZIPs."""
        with zipfile.ZipFile(self.img_zip_path, 'r') as z_img, \
             zipfile.ZipFile(self.label_zip_path, 'r') as z_lab:
            
            all_files = z_img.namelist()
            img_paths = [f for f in all_files if f"leftImg8bit/{self.mode}" in f and f.endswith('.png')]
            
            for path in img_paths:
                # 1. Load the Camera Image
                with z_img.open(path) as f:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
                
                # 2. Derive the Ground Truth path (mapping leftImg8bit to gtFine)
                # Example: leftImg8bit/val/city/file_leftImg8bit.png -> gtFine/val/city/file_gtFine_labelIds.png
                label_path = path.replace("leftImg8bit/", "gtFine/")
                label_path = label_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                
                # 3. Load the Ground Truth Label Mask
                try:
                    with z_lab.open(label_path) as f_lab:
                        gt_mask = np.array(Image.open(io.BytesIO(f_lab.read())))
                except KeyError:
                    gt_mask = None # Fallback if specific label is missing
                
                yield img, gt_mask, path

def get_dataloader():
    if params.DATASET_NAME == "CITYSCAPES":
        return CityscapesZipLoader(mode='val')
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_class = getattr(datasets, params.DATASET_NAME)
        return dataset_class(root=params.DATASET_ROOT, train=False, download=True, transform=transform)