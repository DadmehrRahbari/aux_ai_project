# aux_ai_project/datasets/torchvision_adapter.py

import torchvision.datasets as tv_datasets
import torchvision.transforms as T
from datasets.base import DatasetAdapter

class TorchVisionAdapter(DatasetAdapter):
    """
    Generic TorchVision adapter (image datasets only)
    """

    SUPPORTED_DATASETS = [
        "CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "KMNIST", "SVHN"
    ]

    def __init__(self, dataset_name: str, root: str = "./data", transform=None, download: bool = True):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported TorchVision dataset: {dataset_name}")

        self.dataset_cls = getattr(tv_datasets, dataset_name)
        self.root = root
        self.download = download
        self.transform = transform or T.ToTensor()
        self._datasets = {}

    def _load(self, split: str):
        if split not in self._datasets:
            kwargs = {"root": self.root, "download": self.download, "transform": self.transform}
            # SVHN uses split, others use train
            if self.dataset_cls.__name__ == "SVHN":
                kwargs["split"] = split
            elif "train" in self.dataset_cls.__init__.__code__.co_varnames:
                kwargs["train"] = split == "train"
            self._datasets[split] = self.dataset_cls(**kwargs)
        return self._datasets[split]

    def get_metadata(self):
        ds = self._load("train")
        x, y = ds[0]
        return {
            "modality": "image",
            "input_shape": tuple(x.shape),
            "num_classes": len(set(ds.targets)) if hasattr(ds, "targets") else None,
            "label_type": "categorical",
        }

    def get_split(self, split: str, client_id: int = 0, num_clients: int = 1):
        ds = self._load(split)
        for idx in range(client_id, len(ds), num_clients):
            x, y = ds[idx]
            yield {"input": x, "label": y, "metadata": {"index": idx}}

    def __len__(self):
        return len(self._load("train"))
