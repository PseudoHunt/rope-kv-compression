"""Calibration / held-out token data.

The brief assumed a SALS replication repo with a C4 calibration pipeline; none exists on this
machine, so this module rebuilds it: C4 documents are concatenated (EOS-separated), tokenized with
the Llama-2 tokenizer, and cut into fixed-length chunks.
  - calibration: C4 train shard 00000, first N chunks
  - held-out:    C4 validation shard 00000 (disjoint from calibration by construction)
"""
import glob
import os

import torch

HF = "/home/jl_fs/hf"
MODEL = "/home/jl_fs/models/llama-2-7b-chat"
CACHE = "/home/jl_fs/rope_equiv/cache"


def _c4_file(split):
    pat = f"{HF}/hub/datasets--allenai--c4/snapshots/*/en/c4-{split}.00000-of-*.json.gz"
    return sorted(glob.glob(pat))[0]


def c4_chunks(split, n_seq, seq_len, tokenizer=None):
    path = f"{CACHE}/c4_{split}_{n_seq}x{seq_len}.pt"
    if os.path.exists(path):
        return torch.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = tokenizer or AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("json", data_files=_c4_file(split), split="train", cache_dir=f"{HF}/datasets")
    need = n_seq * seq_len
    ids, i = [], 0
    while len(ids) < need:
        batch = ds[i : i + 1000]["text"]
        for t in tok(batch, add_special_tokens=False)["input_ids"]:
            ids.extend(t + [tok.eos_token_id])
        i += 1000
    out = torch.tensor(ids[:need], dtype=torch.long).view(n_seq, seq_len)
    # Every chunk starts with BOS, matching how the model sees sequences.
    out[:, 0] = tok.bos_token_id
    os.makedirs(CACHE, exist_ok=True)
    torch.save(out, path)
    return out


def calib_tokens(n_seq=512, seq_len=4096):
    return c4_chunks("train", n_seq, seq_len)


def heldout_tokens(n_seq=50, seq_len=4096):
    return c4_chunks("validation", n_seq, seq_len)
