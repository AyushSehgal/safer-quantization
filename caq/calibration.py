import random
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class TokenizedChunkDataset(Dataset):
    """Chunks a list of token IDs into fixed-length sequences."""

    def __init__(self, token_ids: list[int], seq_len: int, num_samples: int):
        self.seq_len = seq_len
        # Build chunks starting at random offsets, seeded for reproducibility
        self.chunks = []
        max_start = len(token_ids) - seq_len
        if max_start <= 0:
            raise ValueError(
                f"Token sequence too short ({len(token_ids)}) for seq_len={seq_len}"
            )
        step = max_start // num_samples
        for i in range(num_samples):
            start = i * step
            self.chunks.append(token_ids[start : start + seq_len])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {"input_ids": ids}  # shape: (seq_len,) — DataLoader adds batch dim


def _load_wikitext2_tokens(tokenizer, split: str) -> list[int]:
    """Loads WikiText-2, concatenates all text, and tokenizes."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encodings["input_ids"][0].tolist()


def get_calibration_loader(
    tokenizer,
    num_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> DataLoader:
    """
    Returns a DataLoader over num_samples chunks of WikiText-2 train text.
    batch_size=1 to keep peak memory manageable during dual-model forward passes.
    """
    random.seed(seed)
    token_ids = _load_wikitext2_tokens(tokenizer, split="train")
    dataset = TokenizedChunkDataset(token_ids, seq_len, num_samples)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def get_wikitext2_test_loader(
    tokenizer,
    seq_len: int = 2048,
    num_samples: int = 128,
) -> DataLoader:
    """Returns a DataLoader over WikiText-2 test split for PPL evaluation."""
    token_ids = _load_wikitext2_tokens(tokenizer, split="test")
    dataset = TokenizedChunkDataset(token_ids, seq_len, num_samples)
    return DataLoader(dataset, batch_size=1, shuffle=False)
