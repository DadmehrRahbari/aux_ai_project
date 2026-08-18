# aux_ai_project/datasets/base.py

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any

class DatasetAdapter(ABC):
    """
    Dataset abstraction compatible with ANY modality and ANY FL SDK.
    """

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Example:
        {
            "modality": "image" | "tabular" | "text" | "sensor" | "multimodal",
            "input_shape": Any,
            "num_classes": int | None,
            "label_type": "categorical" | "binary" | "regression" | None,
        }
        """
        pass

    @abstractmethod
    def get_split(
        self,
        split: str,
        client_id: int = 0,
        num_clients: int = 1,
    ) -> Iterable[Dict[str, Any]]:
        """
        Returns an iterable over samples for a specific FL client.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
