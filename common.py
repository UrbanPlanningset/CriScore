#!/usr/bin/env python3
"""Shared integrity, ranking, evidence, and serialization operations."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()

def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values),method="average")/(len(values)+1)

def rho(target: np.ndarray,prediction: np.ndarray) -> float:
    value=float(spearmanr(target,prediction).statistic)
    if not math.isfinite(value): raise ValueError("SRCC is undefined for constant or non-finite inputs")
    return value

def temporal_views(tokens: np.ndarray) -> dict[str,np.ndarray]:
    weights=np.linspace(-1,1,tokens.shape[1])
    slope=np.einsum("ntd,t->nd",tokens,weights)/(weights*weights).sum()
    return {
        "mean":tokens.mean(1).astype("f4"),
        "stats":np.c_[tokens.mean(1),tokens.std(1),tokens[:,-1]-tokens[:,0],slope].astype("f4"),
    }

def blend(base: np.ndarray,evidence: np.ndarray,beta: float) -> np.ndarray:
    base_rank=rank01(base)
    return (1.0-beta)*base_rank+beta*evidence

def quantile_decode(rank_prediction: np.ndarray,training_target: np.ndarray) -> np.ndarray:
    decoded=np.quantile(training_target,np.clip(rank_prediction,0,1),method="linear")
    return decoded+1e-8*rank_prediction

def write_json(path: Path,payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
