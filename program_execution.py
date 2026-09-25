#!/usr/bin/env python3
"""Executable rank composition and bounded cross-fitted correction."""
from __future__ import annotations

import numpy as np

from common import rank01
from evidence import RidgeLSQR


CORRECTION_ALPHA=100.0


def normalized_weights(program: dict) -> np.ndarray:
    priorities=np.asarray(program["source_weights"],dtype=np.float64)
    if len(priorities)!=len(program["sources"]) or np.any(priorities<0) or priorities.sum()<=0:
        raise ValueError("invalid source priorities")
    temperature=float(program["temperature"])
    if temperature<=0: raise ValueError("temperature must be positive")
    scaled=np.power(priorities,1.0/temperature)
    return scaled/scaled.sum()


def compose_from_ranks(base_rank: np.ndarray,source_rank: dict[str,np.ndarray],program: dict) -> tuple[np.ndarray,np.ndarray]:
    weights=normalized_weights(program)
    evidence=sum(weight*source_rank[source] for weight,source in zip(weights,program["sources"]))
    beta=float(program["beta"])
    if not 0<=beta<=1: raise ValueError("beta must be in [0,1]")
    return (1.0-beta)*base_rank+beta*evidence,evidence


def population_ranks(base: np.ndarray,sources: dict[str,np.ndarray]) -> tuple[np.ndarray,dict[str,np.ndarray]]:
    return rank01(base),{name:rank01(values) for name,values in sources.items()}


def correction_features(composed: np.ndarray,base_rank: np.ndarray,source_rank: dict[str,np.ndarray],program: dict) -> np.ndarray:
    return np.column_stack([composed,base_rank]+[source_rank[source] for source in program["sources"]])


def model_to_dict(model: RidgeLSQR) -> dict:
    return {"alpha":model.alpha,"median":model.median.tolist(),"mean":model.mean.tolist(),
            "scale":model.scale.tolist(),"target_mean":model.target_mean,"coef":model.coef.tolist()}


def model_from_dict(payload: dict) -> RidgeLSQR:
    model=RidgeLSQR(float(payload["alpha"])); model.median=np.asarray(payload["median"],float)
    model.mean=np.asarray(payload["mean"],float); model.scale=np.asarray(payload["scale"],float)
    model.target_mean=float(payload["target_mean"]); model.coef=np.asarray(payload["coef"],float)
    return model


def fit_correction(features: np.ndarray,residual: np.ndarray) -> dict:
    return model_to_dict(RidgeLSQR(CORRECTION_ALPHA).fit(features,residual))


def apply_correction(composed: np.ndarray,features: np.ndarray,program: dict,model_payload: dict | None) -> tuple[np.ndarray,np.ndarray]:
    if not program.get("correction_enabled",False): return composed,np.zeros_like(composed)
    if model_payload is None: raise ValueError("enabled correction requires fitted parameters")
    raw=model_from_dict(model_payload).predict(features)
    bound=float(program["correction_bound"]); strength=float(program["correction_strength"])
    delta=strength*np.clip(raw,-bound,bound)
    return composed+delta,delta
