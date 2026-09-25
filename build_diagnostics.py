#!/usr/bin/env python3
"""Build LLM-facing diagnostics exclusively from the training OOF bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import rho,sha256,write_json
from evidence import fit_predict,select_head_config


SOURCES=("i3d","swin","pose")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()

    bundle=np.load(args.bundle,allow_pickle=False)
    required={"target","fold_id","base_oof"}|{f"{source}_tokens" for source in SOURCES}
    missing=required-set(bundle.files)
    if missing: raise ValueError(f"training bundle missing fields: {sorted(missing)}")
    target=bundle["target"].astype(float); fold_id=bundle["fold_id"].astype(int)
    base=bundle["base_oof"].astype(float); folds=sorted(map(int,np.unique(fold_id)))
    if len(folds)<2: raise ValueError("diagnostics require at least two folds")
    if not np.isfinite(target).all() or not np.isfinite(base).all():
        raise ValueError("training bundle contains non-finite targets or predictions")
    if any(np.sum(fold_id==fold)<2 for fold in folds):
        raise ValueError("each diagnostics fold requires at least two samples")

    base_fold=[rho(target[fold_id==fold],base[fold_id==fold]) for fold in folds]
    diagnostics={}
    for source in SOURCES:
        tokens=bundle[f"{source}_tokens"].astype("f4")
        prediction=np.full(len(target),np.nan); selected_heads=[]
        for held_fold in folds:
            fit_index=np.flatnonzero(fold_id!=held_fold); held_index=np.flatnonzero(fold_id==held_fold)
            config=select_head_config(tokens,target,fold_id,excluded_fold=held_fold)
            prediction[held_index]=fit_predict(tokens,target,fit_index,held_index,config)
            selected_heads.append({"fold":held_fold,"view":config.view,"alpha":config.alpha})
        if not np.isfinite(prediction).all(): raise RuntimeError(f"incomplete {source} diagnostic predictions")
        fold_srcc=[rho(target[fold_id==fold],prediction[fold_id==fold]) for fold in folds]
        diagnostics[source]={"fold_srcc":fold_srcc,"mean_srcc":float(np.mean(fold_srcc)),
                             "worst_fold_gain_over_base":float(np.min(np.asarray(fold_srcc)-base_fold)),
                             "selected_heads":selected_heads}

    write_json(args.output,{"schema":"cirscore-training-diagnostics-v1","training_only":True,
                            "n_train":len(target),"fold_count":len(folds),
                            "base_fold_srcc":base_fold,"sources":diagnostics,
                            "bundle_sha256":sha256(args.bundle)})


if __name__=="__main__": main()
