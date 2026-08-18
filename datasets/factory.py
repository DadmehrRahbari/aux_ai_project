# aux_ai_project/datasets/factory.py

from datasets.torchvision_adapter import TorchVisionAdapter
from datasets.base import DatasetAdapter

class DatasetFactory:
    @staticmethod
    def from_params(params, split="train"):
        """
        Returns a dataset adapter instance based on params.py configuration.
        """
        backend = params.DATASET_BACKEND
        name = params.DATASET_NAME
        root = getattr(params, "DATASET_ROOT", "./data")

        if backend == "torchvision":
            return TorchVisionAdapter(dataset_name=name, root=root)
        elif backend == "custom":
            from datasets.custom_adapter import CustomDatasetAdapter
            return CustomDatasetAdapter(name=name, root=root)
        elif backend == "sdk":
            from datasets.sdk_adapter import SDKDatasetAdapter
            return SDKDatasetAdapter(name=name, root=root)
        else:
            raise ValueError(f"Unknown dataset backend: {backend}")
