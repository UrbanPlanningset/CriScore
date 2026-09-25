#!/usr/bin/env python3
"""Two-member CoRe base-model library used by the paper-aligned freeze stage."""
from __future__ import annotations
import math, random
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

SEED=20260901; EPOCHS=60; VOTERS=10; RANGE=30.0

def rho(y,p):
    value=float(spearmanr(y,p).statistic)
    return value if math.isfinite(value) else -1.0

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Tree(nn.Module):
    def __init__(self,feature_dim: int):
        super().__init__(); hidden=256; self.feature_dim=int(feature_dim)
        if self.feature_dim<=0: raise ValueError("feature_dim must be positive")
        self.first=nn.Sequential(nn.Linear(2*self.feature_dim+1,hidden),nn.ReLU())
        self.fs=nn.ModuleList([nn.Sequential(nn.Conv1d(2**d*hidden,2**d*2*hidden,1,groups=2**d),nn.ReLU()) for d in range(4)])
        self.cs=nn.ModuleList([nn.Conv1d(2**d*hidden,2**d*2,1,groups=2**d) for d in range(4)])
        self.reg=nn.Conv1d(16*hidden,16,1,groups=16)
    def forward(self,z):
        x=self.first(z).unsqueeze(-1); probabilities=[]; batch_size=len(z)
        for classifier,feature_layer in zip(self.cs,self.fs):
            p=torch.log_softmax(classifier(x).squeeze(-1).view(batch_size,-1,2),-1); x=feature_layer(x)
            if probabilities: p=probabilities[-1].view(batch_size,-1,1).expand(-1,-1,2)+p
            probabilities.append(p)
        return probabilities[-1].reshape(batch_size,-1),self.reg(x).squeeze(-1)

def bounds(completeness,difficulty,train_index):
    differences=[]
    for value in np.unique(difficulty[train_index]):
        group=completeness[train_index[difficulty[train_index]==value]]
        for i in range(len(group)): differences.extend(np.abs(group[i+1:]-group[i]))
    differences=sorted(differences); positive=[]
    for i in range(8):
        left=0 if i==0 else differences[int(i/8*(len(differences)-1))]
        right=RANGE if i==7 else differences[int((i+1)/8*(len(differences)-1))]
        positive.append([left,right])
    return np.asarray([[-right,-left] for left,right in positive[::-1]]+positive,np.float32)

def labels(values,boundaries):
    leaf_index=[]; residual=[]
    for value in values:
        chosen=15 if value>=0 else 0
        for index,(left,right) in enumerate(boundaries):
            if (value>=0 and left<=value<right) or (value<0 and left<value<=right): chosen=index; break
        left,right=boundaries[chosen]; leaf_index.append(chosen)
        residual.append((value-left)/(right-left) if right>left else value-left)
    return np.asarray(leaf_index,np.int64),np.asarray(residual,np.float32)

def fit_model(features,completeness,difficulty,train_index,seed,device):
    features=np.asarray(features)
    if features.ndim!=2 or features.shape[1]==0: raise ValueError("base features must have shape [N,D] with D>0")
    if not np.isfinite(features).all(): raise ValueError("base features contain non-finite values")
    seed_all(seed); model=Tree(features.shape[1]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    boundaries=bounds(completeness,difficulty,train_index)
    groups={float(d):train_index[difficulty[train_index]==d] for d in np.unique(difficulty[train_index])}
    for epoch in range(EPOCHS):
        generator=np.random.default_rng(seed+epoch); order=generator.permutation(train_index); exemplars=[]
        for index in order:
            pool=groups[float(difficulty[index])]; alternatives=pool[pool!=index]
            exemplars.append(int(generator.choice(alternatives if len(alternatives) else pool)))
        exemplars=np.asarray(exemplars); first,second=features[order],features[exemplars]
        first_score,second_score=completeness[order],completeness[exemplars]
        x=np.concatenate([np.c_[first,second,first_score/RANGE],np.c_[second,first,second_score/RANGE]]).astype("float32")
        delta=np.r_[second_score-first_score,first_score-second_score]
        leaf_index,residual=labels(delta,boundaries); permutation=generator.permutation(len(x)); model.train()
        for start in range(0,len(x),64):
            batch=permutation[start:start+64]; xb=torch.from_numpy(x[batch]).to(device)
            lb=torch.from_numpy(leaf_index[batch]).to(device); rb=torch.from_numpy(residual[batch]).to(device)
            log_probability,prediction=model(xb)
            loss=nn.functional.nll_loss(log_probability,lb)+nn.functional.mse_loss(prediction.gather(1,lb[:,None]).squeeze(),rb)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model,boundaries,groups

def predict(model,boundaries,groups,features,completeness,difficulty,test_index,seed,device):
    features=np.asarray(features)
    if features.ndim!=2 or features.shape[1]!=model.feature_dim:
        raise ValueError(f"base feature dimension mismatch: expected {model.feature_dim}, got {features.shape[1] if features.ndim==2 else 'non-2D'}")
    train_difficulties=np.asarray(sorted(groups),np.float32); examples=[]; owners=[]; exemplar_scores=[]
    for owner,index in enumerate(test_index):
        value=float(difficulty[index])
        group_value=value if value in groups else float(train_difficulties[np.abs(train_difficulties-value).argmin()])
        pool=groups[group_value]; generator=np.random.default_rng(seed+100000+int(index))
        chosen=generator.choice(pool,VOTERS,replace=len(pool)<VOTERS)
        for exemplar in chosen:
            examples.append(np.r_[features[exemplar],features[index],completeness[exemplar]/RANGE])
            owners.append(owner); exemplar_scores.append(completeness[exemplar])
    examples=np.asarray(examples,np.float32); prediction=np.zeros(len(test_index)); counts=np.zeros(len(test_index)); model.eval()
    with torch.no_grad():
        for start in range(0,len(examples),512):
            log_probability,residual=model(torch.from_numpy(examples[start:start+512]).to(device))
            probability=log_probability.exp().cpu().numpy(); leaf=probability.argmax(1); residual=residual.cpu().numpy()
            delta=np.asarray([boundaries[index,0]+(boundaries[index,1]-boundaries[index,0])*residual[row,index] for row,index in enumerate(leaf)])
            for row,value in enumerate(delta):
                flat_index=start+row; owner=owners[flat_index]
                prediction[owner]+=(value+exemplar_scores[flat_index])*difficulty[test_index[owner]]; counts[owner]+=1
    return prediction/counts
