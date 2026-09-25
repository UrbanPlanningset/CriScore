#!/usr/bin/env python3
"""Strict data-boundary loaders for train labels and label-free evaluation metadata."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import sha256

def load_split(path: Path):
    manifest=json.loads(path.read_text()); train=list(map(str,manifest["train_ids"])); test=list(map(str,manifest["test_ids"]))
    if not train or not test: raise ValueError("nonempty split IDs required")
    if len(train)+len(test)!=int(manifest["dataset_size"]): raise ValueError("dataset_size mismatch")
    if len(set(train+test))!=len(train)+len(test): raise ValueError("duplicate or overlapping IDs")
    if abs(len(train)/(len(train)+len(test))-.8)>.002: raise ValueError("paper protocol requires 80/20")
    if not manifest.get("source_sha256"): raise ValueError("split source hash required")
    return manifest,train,test

def indexed_npz(path: Path,required):
    z=np.load(path,allow_pickle=False)
    missing=set(required)-set(z.files)
    if missing: raise ValueError(f"{path.name} missing fields: {sorted(missing)}")
    ids=list(map(str,z["sample_id"]))
    if len(ids)!=len(set(ids)): raise ValueError(f"{path.name} has duplicate IDs")
    return z,{sample_id:index for index,sample_id in enumerate(ids)}

def ordered(array,lookup,ids):
    missing=[sample_id for sample_id in ids if sample_id not in lookup]
    if missing: raise ValueError(f"missing IDs: {missing[:3]}")
    return array[[lookup[sample_id] for sample_id in ids]]

def load_training_labels(path: Path,train_ids):
    z,lookup=indexed_npz(path,("sample_id","target","difficulty"))
    target=ordered(z["target"],lookup,train_ids).astype(float)
    difficulty=ordered(z["difficulty"],lookup,train_ids).astype(float)
    if not np.isfinite(target).all() or not np.isfinite(difficulty).all(): raise ValueError("training labels contain non-finite values")
    if np.any(difficulty<=0): raise ValueError("difficulty must be strictly positive")
    return target,difficulty

def load_evaluation_metadata(path: Path,test_ids):
    z,lookup=indexed_npz(path,("sample_id","difficulty"))
    if "target" in z.files or "final_score" in z.files:
        raise ValueError("evaluation metadata must not contain labels")
    difficulty=ordered(z["difficulty"],lookup,test_ids).astype(float)
    if not np.isfinite(difficulty).all() or np.any(difficulty<=0):
        raise ValueError("evaluation difficulty must be finite and strictly positive")
    return difficulty

def load_tokens(i3d_path: Path,swin_path: Path,pose_path: Path,ids):
    outputs={}
    for name,path in (("i3d",i3d_path),("swin",swin_path),("pose",pose_path)):
        z,lookup=indexed_npz(path,("sample_id","tokens"))
        values=ordered(z["tokens"],lookup,ids).astype("f4")
        if values.ndim!=3 or values.shape[1]==0 or values.shape[2]==0:
            raise ValueError(f"{name} tokens must have shape [N,T,D] with T,D>0")
        if not np.isfinite(values).all(): raise ValueError(f"{name} tokens contain non-finite values")
        outputs[name]=values
    return outputs
