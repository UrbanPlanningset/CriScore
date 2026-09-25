#!/usr/bin/env python3
"""Training-only evidence-head selection and frozen refitting."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from common import rho,temporal_views

ALPHAS=(10.0,100.0,1000.0)

class RidgeLSQR:
    def __init__(self,alpha: float): self.alpha=float(alpha)
    def fit(self,x,y):
        x=np.asarray(x,dtype=np.float64); self.median=np.nanmedian(x,axis=0)
        x=np.where(np.isfinite(x),x,self.median); self.mean=x.mean(0); self.scale=x.std(0)
        self.scale[self.scale==0]=1.0; z=(x-self.mean)/self.scale
        self.target_mean=float(np.mean(y)); centered=np.asarray(y,dtype=np.float64)-self.target_mean
        dual=np.linalg.solve(z@z.T+self.alpha*np.eye(len(z)),centered); self.coef=z.T@dual
        return self
    def predict(self,x):
        x=np.asarray(x,dtype=np.float64); x=np.where(np.isfinite(x),x,self.median)
        return self.target_mean+((x-self.mean)/self.scale)@self.coef

@dataclass(frozen=True)
class HeadConfig:
    view: str
    alpha: float

def select_head_config(tokens,target,fold_id,excluded_fold=None) -> HeadConfig:
    views=temporal_views(tokens); folds=sorted(map(int,np.unique(fold_id)))
    eligible=np.ones(len(target),dtype=bool) if excluded_fold is None else fold_id!=excluded_fold
    positions=np.flatnonzero(eligible); scored=[]
    for view_name,features in views.items():
        for alpha in ALPHAS:
            prediction=np.full(len(positions),np.nan)
            for fold in folds:
                valid=eligible&(fold_id==fold)
                if not valid.any(): continue
                fit=eligible&(fold_id!=fold)
                model=RidgeLSQR(alpha).fit(features[fit],target[fit])
                local=np.searchsorted(positions,np.flatnonzero(valid))
                prediction[local]=model.predict(features[valid])
            valid=np.isfinite(prediction)
            scored.append((rho(target[positions][valid],prediction[valid]),view_name,alpha))
    _,view_name,alpha=max(scored,key=lambda row:(row[0],row[1],-row[2]))
    return HeadConfig(view_name,float(alpha))

def fit_predict(tokens,target,fit_index,predict_index,config: HeadConfig):
    features=temporal_views(tokens)[config.view]
    return RidgeLSQR(config.alpha).fit(features[fit_index],target[fit_index]).predict(features[predict_index])
