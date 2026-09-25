#!/usr/bin/env python3
"""Create training-only cross-fitted predictions and the program-selection bundle."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import KFold
from common import sha256
from core_model import fit_model,predict
from data_io import load_split,load_tokens,load_training_labels

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",type=Path,required=True); p.add_argument("--train-labels",type=Path,required=True)
    p.add_argument("--i3d",type=Path,required=True); p.add_argument("--swin",type=Path,required=True)
    p.add_argument("--pose",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--folds",type=int,default=5); p.add_argument("--device",default="cuda:0"); args=p.parse_args()
    _,train_ids,_=load_split(args.split); target,difficulty=load_training_labels(args.train_labels,train_ids)
    tokens=load_tokens(args.i3d,args.swin,args.pose,train_ids); base_features=tokens["i3d"].mean(1).astype("f4")
    component=target/difficulty; fold_id=np.full(len(train_ids),-1,dtype=int); base_members=np.full((2,len(train_ids)),np.nan)
    splitter=KFold(args.folds,shuffle=True,random_state=20260901); device=torch.device(args.device)
    for fold,(fit_index,valid_index) in enumerate(splitter.split(np.arange(len(train_ids)))):
        fold_id[valid_index]=fold
        for member,offset in enumerate((11,29)):
            seed=20260900+100*fold+offset
            model,bounds,groups=fit_model(base_features,component,difficulty,fit_index,seed,device)
            base_members[member,valid_index]=predict(model,bounds,groups,base_features,component,difficulty,valid_index,seed,device)
    if not np.isfinite(base_members).all(): raise RuntimeError("incomplete base OOF predictions")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,sample_id=np.asarray(train_ids),target=target,difficulty=difficulty,fold_id=fold_id,
                        base_oof=base_members.mean(0),base_member_oof=base_members,
                        i3d_tokens=tokens["i3d"],swin_tokens=tokens["swin"],pose_tokens=tokens["pose"],
                        split_sha256=sha256(args.split),train_labels_sha256=sha256(args.train_labels))

if __name__=="__main__": main()
