from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class ParamBatch:
    tensor: torch.Tensor
    names:  list[str]

    def __getitem__(self, name: str) -> torch.Tensor:
        idx = self.names.index(name)
        return self.tensor[:, idx]

    def to(self, device) -> ParamBatch:
        return ParamBatch(self.tensor.to(device), self.names)

    @staticmethod
    def from_dict(params: dict[str, float | list], 
                  order:  list[str],
                  device = "cpu") -> ParamBatch:
        """
        Example:
            ParamBatch.from_dict({"alpha": 1.0, "beta": -2.5, "r": 0.13, "G": 0.0},
                                 order=["alpha", "beta", "r", "G"])
            # -> tensor shape (1, 4)

            ParamBatch.from_dict({"alpha": [1.0, 2.0], "beta": [-2.5, -2.5], ...},
                                 order=["alpha", "beta", "r", "G"])
            # -> tensor shape (2, 4)
        """
        cols = []
        for key in order:
            val = params[key]
            if isinstance(val, (int, float)):
                val = [val]
            cols.append(torch.tensor(val, dtype=torch.float32))

        tensor = torch.stack(cols, dim=1)
        return ParamBatch(tensor.to(device), order)