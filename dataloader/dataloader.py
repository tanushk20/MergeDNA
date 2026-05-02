import numpy as np
import torch
from torch.utils.data import Dataset


BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _parse_fasta(path: str) -> list[str]:
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current:
                    sequences.append(''.join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append(''.join(current))
    return sequences


def _one_hot(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = BASE_TO_IDX.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


class DNADataset(Dataset):
    def __init__(self, fasta_path: str):
        self.sequences = _parse_fasta(fasta_path)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = _one_hot(self.sequences[idx])
        return torch.from_numpy(arr)
